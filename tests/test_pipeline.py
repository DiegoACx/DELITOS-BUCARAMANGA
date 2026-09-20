"""Pruebas basicas del pipeline guardado.

Requieren models/pipeline.joblib, que genera `python src/train.py`.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pipeline.joblib"

EXPECTED_CLASSES = {
    "DELITOS CONTRA EL PATRIMONIO ECONOMICO",
    "DELITOS CONTRA LA VIDA Y LA INTEGRIDAD PERSONAL",
    "DELITOS CONTRA LA FAMILIA",
    "DELITOS CONTRA LA LIBERTAD, INTEGRIDAD Y FORMACION SEXUALES",
    "DELITOS CONTRA LA SEGURIDAD PUBLICA",
}

# Dos reportes de ejemplo con valores validos de cada variable de entrada.
SAMPLE = pd.DataFrame(
    [
        {"num_comuna": "3", "mes_num": "6", "sexo": "FEMENINO", "movil_victima": "A PIE", "dia_nombre": "sábado",
         "rango_edad": "ADULTEZ", "rango_horario": "10:00-10:59", "curso_vida": "25-29", "clase_sitio": "VIAS PUBLICAS"},
        {"num_comuna": "1", "mes_num": "12", "sexo": "MASCULINO", "movil_victima": "MOTOCICLETA", "dia_nombre": "lunes",
         "rango_edad": "JUVENTUD", "rango_horario": "20:00-20:59", "curso_vida": "20-24", "clase_sitio": "CASAS DE HABITACION"},
    ]
)


@pytest.fixture(scope="module")
def pipeline():
    if not MODEL_PATH.exists():
        pytest.fail(f"Falta {MODEL_PATH}. Genera el modelo con: python src/train.py")
    return joblib.load(MODEL_PATH)


def test_pipeline_carga_y_predice(pipeline):
    assert hasattr(pipeline, "predict_proba")
    assert hasattr(pipeline, "predict")
    assert list(SAMPLE.columns) == list(pipeline.feature_names_in_)


def test_predict_proba_suma_uno(pipeline):
    proba = pipeline.predict_proba(SAMPLE)
    assert proba.shape == (len(SAMPLE), 5)
    assert np.all(proba >= 0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classes_tiene_las_5_clases(pipeline):
    assert len(pipeline.classes_) == 5
    assert set(pipeline.classes_) == EXPECTED_CLASSES
