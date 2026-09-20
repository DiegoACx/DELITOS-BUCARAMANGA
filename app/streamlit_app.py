"""Clasificador de tipologia de reportes (Streamlit).

Uso:
    streamlit run app/streamlit_app.py

Carga models/pipeline.joblib (generado con `python src/train.py`). Las opciones de cada selector
salen de las categorias que aprendio el OneHotEncoder del pipeline; las etiquetas legibles y el
orden son solo de presentacion.

`rango_edad` y `curso_vida` son dos formas de expresar la edad: la app pide la edad y deriva ambas.
"""
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prep import EDAD_BINS, EDAD_LABELS  # misma regla de rango_edad que usa la limpieza  # noqa: E402

MODEL_PATH = ROOT / "models" / "pipeline.joblib"
TITULO = "Clasificador de tipología de reportes"
AVISO = "Clasifica el tipo de un reporte según su contexto; no mide riesgo"
OTRO_SITIO = "__OTRO__"  # valor no visto en entrenamiento: el pipeline lo agrupa como "infrecuente"

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
EDADES = ["PRIMERA INFANCIA", "INFANCIA", "ADOLESCENCIA", "JUVENTUD", "ADULTEZ", "PERSONA MAYOR"]
COMUNAS = {"1": "Norte", "2": "Nororiental", "3": "San Francisco", "4": "Occidental", "5": "García Rovira",
           "6": "La Concordia", "7": "La Ciudadela", "8": "Suroccidente", "9": "La Pedregosa", "10": "Provenza",
           "11": "Sur", "12": "Cabecera del Llano", "13": "Oriental", "14": "Morrorico", "15": "Centro",
           "16": "Lagos del Cacique", "17": "Mutis"}


def legible(texto: str) -> str:
    """'MOTOCICLETA' -> 'Motocicleta'."""
    return texto.capitalize()


def etiqueta_clase(clase: str) -> str:
    """'DELITOS CONTRA LA FAMILIA' -> 'La familia' (el prefijo comun se explica en un pie de pagina)."""
    return legible(clase.removeprefix("DELITOS CONTRA "))


def derivar_rango_edad(edad: int) -> str:
    """Misma regla que src/prep.py: pd.cut(edad, EDAD_BINS, EDAD_LABELS, right=True, include_lowest=True)."""
    return str(pd.cut([edad], bins=EDAD_BINS, labels=EDAD_LABELS, right=True, include_lowest=True)[0])


def derivar_curso_vida(edad: int, categorias) -> str:
    """Devuelve la categoria de curso_vida ('25-29', '85 o más', ...) que contiene la edad.

    Los limites se leen de las propias categorias del pipeline, no de una tabla escrita a mano.
    """
    for cat in categorias:
        rango = re.fullmatch(r"(\d+)-(\d+)", cat)
        if rango and int(rango[1]) <= edad <= int(rango[2]):
            return cat
        abierto = re.fullmatch(r"(\d+) o más", cat)
        if abierto and edad >= int(abierto[1]):
            return cat
    raise ValueError(f"Ninguna categoria de curso_vida contiene la edad {edad}: {list(categorias)}")


@st.cache_resource
def cargar_pipeline():
    return joblib.load(MODEL_PATH)


def categorias(pipeline) -> dict:
    """Categorias de cada variable segun los OneHotEncoder ajustados del pipeline."""
    pre = pipeline.named_steps["pre"]
    cat = pre.named_transformers_["cat"]
    opciones = {nombre: [str(c) for c in cats] for nombre, cats in zip(cat.feature_names_in_, cat.categories_)}
    sitio = pre.named_transformers_["sitio"]
    infrecuentes = set(sitio.infrequent_categories_[0]) if sitio.infrequent_categories_[0] is not None else set()
    opciones["clase_sitio"] = [str(c) for c in sitio.categories_[0] if c not in infrecuentes]
    return opciones


def main() -> None:
    st.set_page_config(page_title=TITULO)
    st.title(TITULO)
    st.warning(AVISO)

    if not MODEL_PATH.exists():
        st.error(f"Falta {MODEL_PATH.relative_to(ROOT)}. Genéralo con: python src/train.py")
        st.stop()
    pipeline = cargar_pipeline()
    opc = categorias(pipeline)

    comunas = sorted(opc["num_comuna"], key=int)
    meses = sorted(opc["mes_num"], key=int)
    dias = [d for d in DIAS if d in opc["dia_nombre"]] + [d for d in opc["dia_nombre"] if d not in DIAS]
    horas = sorted(opc["rango_horario"], key=lambda s: int(s.split(":")[0]))
    sitios = sorted(opc["clase_sitio"]) + [OTRO_SITIO]

    with st.sidebar:
        st.header("Contexto del reporte")
        comuna = st.selectbox("Comuna", comunas, key="comuna", format_func=lambda n: COMUNAS.get(n, f"Comuna {n}"))
        mes = st.selectbox("Mes", meses, key="mes", format_func=lambda n: MESES[int(n) - 1])
        dia = st.selectbox("Día de la semana", dias, key="dia", format_func=legible)
        franja = st.selectbox("Franja horaria", horas, key="franja")
        sitio = st.selectbox("Lugar del hecho", sitios, key="sitio",
                             format_func=lambda s: "Otro lugar (poco frecuente)" if s == OTRO_SITIO else legible(s))
        sexo = st.selectbox("Sexo de la víctima", sorted(opc["sexo"]), key="sexo", format_func=legible)
        movil = st.selectbox("Cómo se movilizaba la víctima", sorted(opc["movil_victima"]), key="movil",
                             format_func=legible)
        edad = st.number_input("Edad de la víctima (años)", min_value=0, max_value=100, value=30, step=1, key="edad")
        rango_edad = derivar_rango_edad(int(edad))
        curso_vida = derivar_curso_vida(int(edad), opc["curso_vida"])
        st.caption(f"Se deriva de la edad: rango de edad «{legible(rango_edad)}» · curso de vida «{curso_vida}».")

    fila = {"num_comuna": comuna, "mes_num": mes, "sexo": sexo, "movil_victima": movil, "dia_nombre": dia,
            "rango_edad": rango_edad, "rango_horario": franja, "curso_vida": curso_vida, "clase_sitio": sitio}
    X = pd.DataFrame([fila])[list(pipeline.feature_names_in_)]
    proba = pipeline.predict_proba(X)[0]

    resultado = pd.DataFrame({"Tipo": [etiqueta_clase(c) for c in pipeline.classes_], "Probabilidad": proba})
    mejor = resultado.loc[resultado["Probabilidad"].idxmax()]
    st.subheader("Probabilidad por tipo de delito")
    st.success(f"Tipo más probable: **{mejor['Tipo']}** ({mejor['Probabilidad']:.1%})")
    st.bar_chart(resultado, x="Tipo", y="Probabilidad", horizontal=True, sort="-Probabilidad")
    st.caption("Cada tipo corresponde a «Delitos contra …» (nombres tomados de las clases del modelo).")


main()
