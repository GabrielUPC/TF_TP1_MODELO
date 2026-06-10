import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    XGBOOST_DISPONIBLE = True
except (ImportError, OSError):
    XGBClassifier = None
    XGBOOST_DISPONIBLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_modelo_ipress.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "modelo_ipress.joblib"
METRICAS_PATH = MODELS_DIR / "metricas_modelo.csv"
CLASES_PATH = MODELS_DIR / "clases_riesgo.json"

VARIABLE_OBJETIVO = "nivel_riesgo_codificado"
CLASES_RIESGO = {0: "bajo", 1: "medio", 2: "alto"}
COLUMNAS_NO_PREDICTORAS = [
    "nivel_riesgo",
    "nivel_riesgo_codificado",
    "nombre_ipress",
]
COLUMNAS_IDENTIFICADOR = [
    "ubigeo",
    "codigo_ipress",
    "id_hospitalizacion",
]


def cargar_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset procesado: {DATASET_PATH}. "
            "Ejecute primero: py src/preparar_dataset.py"
        )

    df = pd.read_csv(
        DATASET_PATH,
        dtype={columna: "string" for columna in COLUMNAS_IDENTIFICADOR},
    )
    if VARIABLE_OBJETIVO not in df.columns:
        raise ValueError(
            f"El dataset no contiene la variable objetivo '{VARIABLE_OBJETIVO}'."
        )
    if df.empty:
        raise ValueError("El dataset procesado está vacío.")
    return df


def separar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=COLUMNAS_NO_PREDICTORAS, errors="ignore")
    y = pd.to_numeric(df[VARIABLE_OBJETIVO], errors="raise").astype(int)

    clases_encontradas = set(y.unique())
    clases_esperadas = set(CLASES_RIESGO)
    if clases_encontradas != clases_esperadas:
        raise ValueError(
            "La variable objetivo debe contener las clases 0, 1 y 2. "
            f"Clases encontradas: {sorted(clases_encontradas)}"
        )
    if y.value_counts().min() < 2:
        raise ValueError(
            "Cada clase necesita al menos dos registros para aplicar "
            "una partición estratificada."
        )
    return X, y


def crear_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    columnas_categoricas = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()

    columnas_sin_tipo = sorted(
        set(X.columns).difference(columnas_categoricas + columnas_numericas)
    )
    if columnas_sin_tipo:
        raise ValueError(
            "No se pudo determinar el tipo de estas columnas: "
            + ", ".join(columnas_sin_tipo)
        )

    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                columnas_categoricas,
            ),
            ("numericas", StandardScaler(), columnas_numericas),
        ],
        remainder="drop",
    )


def obtener_modelos() -> dict[str, object]:
    modelos: dict[str, object] = {
        "Regresion_Logistica": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBOOST_DISPONIBLE and XGBClassifier is not None:
        modelos["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

    return modelos


def evaluar_modelo(
    nombre: str,
    modelo: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    y_pred = modelo.predict(X_test)
    etiquetas = list(CLASES_RIESGO)
    nombres = [CLASES_RIESGO[clase] for clase in etiquetas]

    metricas: dict[str, float | str] = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
    }

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(X_test)
        metricas["probabilidad_maxima_promedio"] = float(
            probabilidades.max(axis=1).mean()
        )
    else:
        metricas["probabilidad_maxima_promedio"] = np.nan

    print("\n" + "=" * 70)
    print(f"MODELO: {nombre}")
    print("=" * 70)
    print("Matriz de confusión (bajo, medio, alto):")
    print(confusion_matrix(y_test, y_pred, labels=etiquetas))
    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=etiquetas,
            target_names=nombres,
            zero_division=0,
        )
    )
    print("Métricas:")
    for metrica, valor in metricas.items():
        print(f"  {metrica}: {valor}")

    return metricas


def guardar_clases() -> None:
    contenido = {str(clase): nombre for clase, nombre in CLASES_RIESGO.items()}
    CLASES_PATH.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def entrenar_modelos() -> Pipeline:
    print("Cargando dataset procesado...")
    df = cargar_dataset()
    X, y = separar_variables(df)

    print(f"Filas: {df.shape[0]}")
    print(f"Columnas predictoras: {X.shape[1]}")
    print("Distribución de la variable objetivo:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    resultados: list[dict[str, float | str]] = []
    modelos_entrenados: list[str] = []
    mejor_modelo: Pipeline | None = None
    mejor_nombre = ""
    mejor_f1 = -1.0

    for nombre, algoritmo in obtener_modelos().items():
        pipeline = Pipeline(
            steps=[
                ("preprocesador", crear_preprocesador(X)),
                ("modelo", algoritmo),
            ]
        )
        print(f"\nEntrenando modelo: {nombre}...")

        try:
            pipeline.fit(X_train, y_train)
            metricas = evaluar_modelo(nombre, pipeline, X_test, y_test)
        except Exception as error:
            print(f"No se pudo entrenar {nombre}: {error}")
            continue

        resultados.append(metricas)
        modelos_entrenados.append(nombre)

        f1_actual = float(metricas["f1_macro"])
        if f1_actual > mejor_f1:
            mejor_f1 = f1_actual
            mejor_modelo = pipeline
            mejor_nombre = nombre

    if mejor_modelo is None:
        raise RuntimeError("Ninguno de los modelos pudo entrenarse correctamente.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor_modelo, MODEL_PATH)
    pd.DataFrame(resultados).sort_values(
        "f1_macro", ascending=False
    ).to_csv(METRICAS_PATH, index=False, encoding="utf-8-sig")
    guardar_clases()

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 70)
    print(f"Modelos entrenados: {', '.join(modelos_entrenados)}")
    print(f"Mejor modelo: {mejor_nombre}")
    print(f"Mejor F1 macro: {mejor_f1:.6f}")
    print(f"Modelo guardado: {MODEL_PATH}")
    print(f"Métricas guardadas: {METRICAS_PATH}")
    print(f"Clases guardadas: {CLASES_PATH}")
    return mejor_modelo


if __name__ == "__main__":
    try:
        entrenar_modelos()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error durante el entrenamiento: {error}") from error
