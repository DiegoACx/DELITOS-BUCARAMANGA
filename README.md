<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<h1 align="center">Clasificador de tipología de reportes de delitos en Bucaramanga</h1>

<p align="center">Clasificador validado con corte temporal del tipo de reporte de delito, construido sobre datos abiertos de datos.gov.co.</p>

<p align="center">🇬🇧 <a href="README.en.md">Read in English</a></p>

## Acerca del proyecto

Este proyecto clasifica el **tipo de un reporte de delito** en Bucaramanga (5 clases del campo `tipologia`) a partir de su contexto: comuna, mes, día, franja horaria, lugar, y sexo, edad y movilidad de la víctima. Usa el dataset abierto `x46e-abhz` de [datos.gov.co](https://www.datos.gov.co) con un pipeline reproducible de scikit-learn.

Descarga y limpia los datos, entrena con 2016-2021, prueba con 2022 y guarda un modelo y un reporte de métricas. Incluye además un notebook descriptivo y una app en Streamlit que muestra la probabilidad de cada clase.

### Lo que NO hace

- **No mide riesgo** ni predice delitos futuros, ni cuándo o dónde ocurrirán.
- No evalúa a una persona ni a una zona. Todas las filas son delitos ya reportados: no hay casos "sin delito", así que no se puede estimar riesgo.

La app lo dice: *"Clasifica el tipo de un reporte según su contexto; no mide riesgo"*.

## Datos

- **Fuente:** recurso `x46e-abhz` de datos.gov.co, consultado por su API SODA.
- **Descarga:** `python src/download_data.py` trae los registros con `fecha_hecho < '2023-01-01'`, ordenados por `:id`, en páginas de 20.000 filas, y comprueba el conteo contra el del servidor. Escribe `data/delitos_bucaramanga.csv` (ignorado por git).
- **Por qué se excluye 2023:** en la versión actual del dataset, el 93,4 % de `movil_victima` y el 83,4 % de `edad` figuran como "NO DISPONIBLE" en 2023 (consulta agregada a la API, ver el notebook). La limpieza elimina esas filas y 2023 casi desaparece.
- **Tamaño:** 88.627 filas y 26 columnas descargadas, 76.385 tras limpiar (86,2 %).
- **SHA-256** del archivo descargado: `74aacd380a3c66ddd02d9df7e3754dca772f43496265ba456f1440d588e2d424`. Vale para el recurso tal como estaba el 19/09/2026; el portal puede modificarlo.

## Metodología

- **Limpieza** (`src/prep.py`): los mismos filtros, en el mismo orden, del notebook original (duplicados, armas poco frecuentes, corregimientos y barrios fuera de Bucaramanga, valores "NO DISPONIBLE", edades inválidas). Los mayores descartes son sexo no disponible (8.350 filas) y barrios fuera de Bucaramanga (2.550 filas).
- **Entradas (9):** `num_comuna` (como categoría), `mes_num`, `sexo`, `movil_victima`, `dia_nombre`, `rango_edad`, `rango_horario`, `curso_vida` y `clase_sitio`. Los 224 valores de `clase_sitio` se agrupan en los 19 más frecuentes más una categoría "infrecuente", aprendida solo con entrenamiento. No se usan `arma_empleada` ni `movil_agresor`.
- **Pipeline:** `ColumnTransformer` con `OneHotEncoder` dentro de un `Pipeline` de scikit-learn, ajustado solo con el conjunto de entrenamiento. Todo es categórico, así que no hay escalado.
- **Corte temporal:** entrenamiento 2016-2021 (62.483 filas), prueba 2022 (13.902 filas).
- **Modelos:** `Dummy(prior)`, `LogisticRegression(max_iter=1000)` y `HistGradientBoostingClassifier`. Semilla `random_state=0`, hiperparámetros por defecto, sin ajustar. El modelo guardado es el HGB entrenado solo con 2016-2021.
- **Versiones:** Python 3.13.3, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0.

## Resultados (prueba: 2022)

| Modelo | Exactitud | Macro-F1 | Log loss |
|---|---:|---:|---:|
| Dummy (prior) | 0,56 | 0,14 | 1,35 |
| Regresión logística | 0,61 | 0,41 | 1,07 |
| HistGradientBoosting | 0,62 | 0,42 | 1,07 |

Frente al baseline, el HGB gana unos 5,5 puntos de exactitud y reduce el log loss un 21 % (1,3525 → 1,0705, calculado con `reports/metrics.json`). La mejora es real, pero modesta.

HistGradientBoosting por clase:

| Clase | Precisión | Recall | Filas de prueba |
|---|---:|---:|---:|
| Patrimonio económico | 0,67 | 0,90 | 7.826 |
| Vida e integridad personal | 0,39 | 0,25 | 2.501 |
| Familia | 0,51 | 0,45 | 1.498 |
| Seguridad pública | 0,14 | 0,00 (0,0006) | 1.694 |
| Libertad e integridad sexual | 0,52 | 0,56 | 383 |

Distribución de clases, entrenamiento → prueba: Patrimonio 56,45 → 56,29 %, Vida 20,45 → 17,99 %, Familia 16,70 → 10,78 %, Sexuales 4,20 → 2,75 %, Seguridad pública 2,20 → 12,19 %.

## Limitaciones

- **Una sola división temporal.** No hay validación cruzada ni intervalos de confianza en este repositorio.
- **El modelo se eligió mirando el conjunto de prueba de 2022**, por lo que ese conjunto no es independiente. `SELECTED_MODEL` está fijo en `src/train.py`.
- **`Seguridad pública` casi nunca se predice** (recall 0,0006 con HGB, 0,00 con LR), aunque es el 12,19 % de la prueba.
- **Deriva de etiquetas entre años.** `Seguridad pública` no existe antes de 2021 y casi solo contiene amenazas (3.365 de 3.369 reportes). En 2021 aparecen también lesiones culposas y homicidio culposo en accidente de tránsito. El peso de `Patrimonio` varía entre 50,3 % y 65,8 % según el año. Un modelo entrenado hoy no es confiable hacia adelante.
- **Las variables más influyentes son atributos del propio reporte.** En una evaluación preliminar (no versionada aquí), `clase_sitio` y la edad fueron las de mayor peso. No se conocen antes del hecho, así que esto es clasificar un reporte, no predecir un delito.
- **Sesgo de selección y subregistro.** Se eliminan las filas con sexo o edad no disponibles (cerca del 7-13 % por año), y en los datos solo existen delitos reportados.
- **Atributos demográficos de las víctimas** (sexo, edad) son entradas del modelo. Úsalo como experimento, no para decisiones sobre personas.
- **El dataset cambió.** El notebook original de 2024 usaba 100.993 filas; hoy el recurso tiene 130.202 y 2023 tiene otra codificación, así que esas cifras no son comparables.
- No hay `LICENSE` ni CI, y el devcontainer no se ha construido ni ejecutado (ver abajo).

## Estructura del proyecto

```
.
├─ app/streamlit_app.py
├─ notebooks/01_exploracion.ipynb
├─ models/pipeline.joblib
├─ reports/metrics.json
├─ src/{download_data,prep,train}.py
├─ tests/test_pipeline.py
├─ .devcontainer/devcontainer.json
├─ requirements.txt · requirements-dev.txt · .gitignore
└─ data/                      (generado, ignorado por git)
```

## Cómo correrlo

Requiere Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python src/download_data.py        # crea data/delitos_bucaramanga.csv
python src/train.py                # sobrescribe models/pipeline.joblib y reports/metrics.json
pytest
streamlit run app/streamlit_app.py
```

Los tests solo necesitan `models/pipeline.joblib`, que ya está en el repositorio; no requieren `data/`.

> **Seguridad:** `models/pipeline.joblib` es un pickle de joblib. Cargar un pickle de una fuente desconocida puede ejecutar código arbitrario. Carga solo modelos de confianza y con las versiones fijadas en `requirements.txt`.

## Notebook de exploración

`notebooks/01_exploracion.ipynb` es descriptivo y muestra solo agregados (sin filas de víctimas):

- Tamaño y embudo de limpieza (88.627 → 76.385).
- Tipología por año: `Patrimonio` es la mayoritaria todos los años; `Familia` baja de 20,4 % (2016) a 10,8 % (2022).
- Comunas: cinco (Oriental, San Francisco, Centro, Cabecera del Llano y Norte) reúnen el 53,5 % de los reportes.
- Día y hora: domingo y noche desplazan la mezcla hacia `Vida e integridad personal` (31,5 % y 27,2 %).
- Datos faltantes por año, incluido el problema de 2023 (requiere internet para la consulta agregada a la API).
- Deriva de etiquetas entre entrenamiento y prueba.

## Dev Container

La imagen es `mcr.microsoft.com/devcontainers/python:1-3.13-bookworm`. Al crear el contenedor instala `requirements-dev.txt` y lanza la app en el puerto 8501. Las protecciones CORS y XSRF quedan con sus valores por defecto.

**No probado:** el contenedor nunca se construyó ni se ejecutó. Lo que sí se verificó: que la etiqueta de la imagen existe, que los 53 paquetes resuelven a ruedas `cp313` para Linux (x86_64 y aarch64) con las versiones fijadas, y que un entorno virtual limpio pasa los tests. Un posible problema es que `streamlit`, instalado con `--user`, no quede en el `PATH` del contenedor.

## Historia del proyecto

La primera versión (septiembre de 2024) fue un proyecto académico. En una revisión de septiembre de 2026 se encontró que el notebook original tenía fuga de datos en la evaluación del RandomForest, y que el modelo guardado (una `LinearRegression`) no correspondía a la app (que llamaba `predict_proba`). El repositorio se reconstruyó con corte temporal, baselines honestos y un pipeline reproducible. Los archivos antiguos siguen en el historial de git.

## Autor

- Diego Castro — [@DiegoACx](https://github.com/DiegoACx)

El refactor de 2026 se desarrolló con asistencia de Claude (Anthropic).
