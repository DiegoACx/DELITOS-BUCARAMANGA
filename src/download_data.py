"""Descarga reproducible del recurso x46e-abhz de datos.gov.co (delitos en Bucaramanga).

Uso:
    python src/download_data.py

Escribe data/delitos_bucaramanga.csv (carpeta ignorada por git), e imprime el numero de
filas y el SHA-256 del archivo.

Reproducibilidad
----------------
El recurso sigue creciendo (100.993 filas en septiembre de 2024; 130.202 al verificarlo el
2026-09-19), asi que descargar "todo" o usar solo $limit da resultados distintos con el
tiempo. Por eso la consulta fija:

* $where = fecha_hecho < '2023-01-01'   (solo 2016-2022; en esa fecha son 88.627 filas)
* $order = :id                           (identificador unico de fila -> orden total y estable)
* paginacion con $limit/$offset, para no depender de un limite unico.

Al final se compara el numero de filas descargadas con el conteo del servidor para el mismo
filtro, de modo que una descarga truncada falle en lugar de pasar desapercibida.

Por que se excluye 2023
-----------------------
En la version actual del recurso, 2023 tiene "NO DISPONIBLE" en el 93 % de `movil_victima` y
en el 83 % de `edad`. La limpieza del proyecto elimina esas filas, y de las 12.373 filas de
2023 solo sobreviven 761, todas de una sola clase (VIDA Y LA INTEGRIDAD PERSONAL). Con ellas no
se puede evaluar nada. Ademas, la tipologia SEGURIDAD PUBLICA existe solo en 2021-2022.
"""
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

RESOURCE = "https://www.datos.gov.co/resource/x46e-abhz"
WHERE = "fecha_hecho < '2023-01-01'"
ORDER = ":id"
PAGE_SIZE = 20000
EXPECTED_COLUMNS = 26

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "delitos_bucaramanga.csv"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as resp:
        return resp.read()


def _count_rows_on_server() -> int:
    query = urllib.parse.urlencode({"$select": "count(*)", "$where": WHERE})
    return int(json.loads(_get(f"{RESOURCE}.json?{query}"))[0]["count"])


def _page(offset: int) -> bytes:
    query = urllib.parse.urlencode({"$where": WHERE, "$order": ORDER, "$limit": PAGE_SIZE, "$offset": offset})
    return _get(f"{RESOURCE}.csv?{query}")


def _count_csv_rows(data: bytes) -> int:
    """Filas de datos (sin encabezado); csv.reader respeta saltos de linea dentro de comillas."""
    return sum(1 for _ in csv.reader(io.StringIO(data.decode("utf-8")))) - 1


def main() -> None:
    expected = _count_rows_on_server()
    print(f"Filas en el servidor con {WHERE!r}: {expected:,}")

    chunks, header, offset = [], None, 0
    while True:
        raw = _page(offset)
        first_line, _, body = raw.partition(b"\n")
        if header is None:
            header = first_line
            n_cols = len(next(csv.reader(io.StringIO(header.decode("utf-8")))))
            assert n_cols == EXPECTED_COLUMNS, f"Se esperaban {EXPECTED_COLUMNS} columnas y hay {n_cols}"
            chunks.append(raw if raw.endswith(b"\n") else raw + b"\n")
        else:
            assert first_line == header, "El encabezado cambio entre paginas"
            if body:
                chunks.append(body if body.endswith(b"\n") else body + b"\n")
        n_page = _count_csv_rows(raw)
        print(f"  pagina offset={offset:>6}: {n_page:,} filas")
        if n_page < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    data = b"".join(chunks)
    n_rows = _count_csv_rows(data)
    if n_rows != expected:
        raise SystemExit(f"Descarga incompleta: {n_rows:,} filas descargadas, el servidor reporta {expected:,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"\nArchivo: {OUT.relative_to(ROOT)}")
    print(f"Filas:   {n_rows:,}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
