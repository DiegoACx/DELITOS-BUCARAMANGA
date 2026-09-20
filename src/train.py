"""Entrena y evalua un clasificador de `tipologia` con division temporal y sin fuga de datos.

Uso:
    python src/download_data.py   # una vez, genera data/delitos_bucaramanga.csv
    python src/train.py

Disenio (el mismo que se valido en la evaluacion previa):
* Entrenamiento con 2016-2021 y prueba con 2022 (division temporal).
* Todos los codificadores viven dentro de un Pipeline con ColumnTransformer y se ajustan
  solo con train. `clase_sitio` (224 valores) se agrupa en las 19 mas frecuentes mas una
  categoria "infrecuente", tambien aprendida solo con train.
* Modelos: Dummy(prior), LogisticRegression y HistGradientBoostingClassifier, con semilla
  fija y los hiperparametros por defecto de scikit-learn (sin ajustar).

Salidas:
* models/pipeline.joblib  -> pipeline HistGradientBoosting entrenado solo con 2016-2021.
* reports/metrics.json    -> metricas en test, distribucion de clases, versiones y SHA-256 de los datos.
"""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from prep import limpiar

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "delitos_bucaramanga.csv"
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

TARGET = "tipologia"
CATS = ["num_comuna", "mes_num", "sexo", "movil_victima", "dia_nombre", "rango_edad", "rango_horario", "curso_vida"]
SITIO = "clase_sitio"
FEATURES = CATS + [SITIO]
SEED = 0
TRAIN_YEARS = range(2016, 2022)  # 2016-2021
TEST_YEAR = 2022
SELECTED_MODEL = "HistGradientBoosting"


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
            # 224 valores -> 19 mas frecuentes + "infrecuente" (max_categories=20), aprendido solo con train
            ("sitio", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False), [SITIO]),
        ],
        sparse_threshold=0,
    )


def make_models() -> dict:
    def pipe(clf):
        return Pipeline([("pre", make_preprocessor()), ("clf", clf)])

    return {
        "Dummy(prior)": pipe(DummyClassifier(strategy="prior")),
        "LogisticRegression": pipe(LogisticRegression(max_iter=1000)),
        "HistGradientBoosting": pipe(HistGradientBoostingClassifier(random_state=SEED)),
    }


def class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts()
    return {c: {"n": int(n), "pct": round(n / len(y) * 100, 2)} for c, n in counts.items()}


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)
    classes = model.classes_
    pred = classes[proba.argmax(axis=1)]
    report = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_test, proba, labels=classes)), 4),
        "per_class": {
            c: {
                "precision": round(float(report[c]["precision"]), 4),
                "recall": round(float(report[c]["recall"]), 4),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"No existe {DATA_PATH}. Ejecuta primero: python src/download_data.py")
    data_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    df, steps = limpiar(DATA_PATH)
    train = df[df["anio"].isin(TRAIN_YEARS)]
    test = df[df["anio"] == TEST_YEAR]
    X_train, y_train = train[FEATURES].astype(str), train[TARGET]
    X_test, y_test = test[FEATURES].astype(str), test[TARGET]
    print(f"Datos: {DATA_PATH.name}  SHA-256 {data_sha256}")
    print(f"Filas: {steps[0][1]:,} descargadas -> {len(df):,} limpias | train {min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}: {len(train):,} | test {TEST_YEAR}: {len(test):,}")

    dist_train, dist_test = class_distribution(y_train), class_distribution(y_test)
    print("\nDistribucion de clases (n / %):")
    for c in dist_train:
        print(f"  {c:<62} train {dist_train[c]['n']:>6,} ({dist_train[c]['pct']:>5.1f}%)   test {dist_test[c]['n']:>6,} ({dist_test[c]['pct']:>5.1f}%)")

    models, results = make_models(), {}
    print("\nResultados en test:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        r = results[name]
        print(f"  {name:<22} accuracy={r['accuracy']:.4f}  macro-F1={r['macro_f1']:.4f}  log_loss={r['log_loss']:.4f}")
    print(f"\nPrecision / recall por clase ({SELECTED_MODEL}):")
    for c, m in results[SELECTED_MODEL]["per_class"].items():
        print(f"  {c:<62} P={m['precision']:.3f}  R={m['recall']:.3f}  (n={m['support']:,})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[SELECTED_MODEL], MODEL_PATH, compress=3)

    metrics = {
        "data": {
            "file": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": data_sha256,
            "rows_downloaded": steps[0][1],
            "rows_clean": len(df),
            "cleaning_steps": [{"step": s, "rows": n} for s, n in steps],
        },
        "split": {
            "type": "temporal",
            "train_years": f"{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}",
            "test_year": TEST_YEAR,
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "features": {"categorical": CATS, "clase_sitio": "19 categorias mas frecuentes + 'infrecuente' (aprendido solo con train)"},
        "class_distribution": {"train": dist_train, "test": dist_test},
        "models": results,
        "saved_model": {"name": SELECTED_MODEL, "path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"), "trained_on": f"{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}"},
        "hyperparameters": "defaults de scikit-learn, sin ajustar; random_state=%d" % SEED,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nGuardado: {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) y {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
