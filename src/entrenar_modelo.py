import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
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
METRICAS_ALEATORIO_PATH = MODELS_DIR / "metricas_modelo_aleatorio.csv"
METRICAS_TEMPORAL_PATH = MODELS_DIR / "metricas_modelo_temporal.csv"
METRICAS_COMPLETO_PATH = MODELS_DIR / "metricas_modelo_completo.csv"
METRICAS_INTERPRETABLE_PATH = MODELS_DIR / "metricas_modelo_interpretable.csv"
CLASES_PATH = MODELS_DIR / "clases_riesgo.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
IMPORTANCIA_PATH = MODELS_DIR / "importancia_variables.csv"
IMPORTANCIA_COMPLETO_PATH = MODELS_DIR / "importancia_variables_completo.csv"
IMPORTANCIA_INTERPRETABLE_PATH = (
    MODELS_DIR / "importancia_variables_interpretable.csv"
)

VARIABLE_OBJETIVO = "nivel_riesgo_codificado"
CLASES_RIESGO = {0: "bajo", 1: "medio", 2: "alto"}
MODOS = ("completo", "interpretable")

EXCLUIDAS_COMPLETO = [
    "nivel_riesgo",
    "nivel_riesgo_codificado",
    "nombre_ipress",
    "archivo_origen",
]
EXCLUIDAS_INTERPRETABLE = EXCLUIDAS_COMPLETO + [
    "codigo_ipress",
    "ubigeo",
    "id_hospitalizacion",
]
COLUMNAS_IDENTIFICADOR = [
    "ubigeo",
    "codigo_ipress",
    "id_hospitalizacion",
    "archivo_origen",
]

F1_MINIMO_INTERPRETABLE = 0.70
CAIDA_MAXIMA_INTERPRETABLE = 0.10
ADVERTENCIA_METODOLOGICA = (
    "La variable objetivo se construye mediante reglas aplicadas a los "
    "indicadores del mismo mes y esos indicadores también se usan como "
    "predictores. Por ello, métricas muy altas miden principalmente la "
    "capacidad del modelo para reproducir esas reglas; no demuestran por sí "
    "solas capacidad de pronóstico futuro. Para validar capacidad predictiva "
    "real se requiere evaluación temporal y, de ser posible, etiquetas "
    "observadas en periodos posteriores."
)


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
    if df.empty:
        raise ValueError("El dataset procesado está vacío.")
    if VARIABLE_OBJETIVO not in df.columns:
        raise ValueError(
            f"El dataset no contiene la variable objetivo '{VARIABLE_OBJETIVO}'."
        )
    if "anio" not in df.columns:
        raise ValueError("El dataset no contiene la columna temporal 'anio'.")
    return df


def columnas_excluidas(modo: str) -> list[str]:
    if modo == "completo":
        return EXCLUIDAS_COMPLETO.copy()
    if modo == "interpretable":
        return EXCLUIDAS_INTERPRETABLE.copy()
    raise ValueError(f"Modo de entrenamiento desconocido: {modo}")


def separar_variables(
    df: pd.DataFrame,
    modo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    excluidas = columnas_excluidas(modo)
    X = df.drop(columns=excluidas, errors="ignore")
    y = pd.to_numeric(df[VARIABLE_OBJETIVO], errors="raise").astype(int)

    clases_encontradas = set(y.unique())
    if clases_encontradas != set(CLASES_RIESGO):
        raise ValueError(
            "La variable objetivo debe contener las clases 0, 1 y 2. "
            f"Clases encontradas: {sorted(clases_encontradas)}"
        )
    if y.value_counts().min() < 2:
        raise ValueError(
            "Cada clase necesita al menos dos registros para la evaluación."
        )
    return X, y


def crear_particion_temporal(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int] | None:
    anios = sorted(pd.to_numeric(df["anio"], errors="raise").astype(int).unique())
    if len(anios) < 2:
        print(
            "ADVERTENCIA: solo hay un año disponible; se omite la evaluación "
            "temporal."
        )
        return None

    anio_prueba = anios[-1]
    mascara_prueba = df["anio"].astype(int).eq(anio_prueba)
    X_train = X.loc[~mascara_prueba]
    X_test = X.loc[mascara_prueba]
    y_train = y.loc[~mascara_prueba]
    y_test = y.loc[mascara_prueba]

    if X_train.empty or X_test.empty:
        print(
            "ADVERTENCIA: la partición temporal quedó vacía; se omite esta "
            "evaluación."
        )
        return None
    if set(y_train.unique()) != set(CLASES_RIESGO):
        print(
            "ADVERTENCIA: el entrenamiento temporal no contiene las tres "
            "clases; se omite esta evaluación."
        )
        return None

    return X_train, X_test, y_train, y_test, anio_prueba


def crear_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    categoricas = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    sin_tipo = sorted(set(X.columns).difference(categoricas + numericas))
    if sin_tipo:
        raise ValueError(
            "No se pudo determinar el tipo de estas columnas: "
            + ", ".join(sin_tipo)
        )

    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                categoricas,
            ),
            ("numericas", StandardScaler(), numericas),
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


def crear_pipeline(X: pd.DataFrame, algoritmo: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocesador", crear_preprocesador(X)),
            ("modelo", clone(algoritmo)),
        ]
    )


def evaluar_modelo(
    nombre: str,
    modo: str,
    tipo_evaluacion: str,
    modelo: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    anio_prueba: int | None = None,
) -> dict[str, float | int | str | None]:
    y_pred = modelo.predict(X_test)
    etiquetas = list(CLASES_RIESGO)
    nombres = [CLASES_RIESGO[clase] for clase in etiquetas]

    metricas: dict[str, float | int | str | None] = {
        "modo": modo,
        "evaluacion": tipo_evaluacion,
        "modelo": nombre,
        "anio_prueba": anio_prueba,
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
    probabilidades = modelo.predict_proba(X_test)
    metricas["probabilidad_maxima_promedio"] = float(
        probabilidades.max(axis=1).mean()
    )

    print("\n" + "=" * 74)
    print(
        f"MODO: {modo} | EVALUACIÓN: {tipo_evaluacion} | MODELO: {nombre}"
    )
    print("=" * 74)
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
    return metricas


def entrenar_modo(
    df: pd.DataFrame,
    modo: str,
    algoritmos: dict[str, object],
) -> dict:
    X, y = separar_variables(df, modo)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    resultados_aleatorios = []
    resultados_temporales = []
    mejor_nombre = ""
    mejor_f1 = -1.0
    temporal = crear_particion_temporal(df, X, y)

    for nombre, algoritmo in algoritmos.items():
        print(f"\nEntrenando {nombre} en modo {modo}...")
        modelo_aleatorio = crear_pipeline(X, algoritmo)
        modelo_aleatorio.fit(X_train, y_train)
        metricas_aleatorias = evaluar_modelo(
            nombre,
            modo,
            "aleatoria",
            modelo_aleatorio,
            X_test,
            y_test,
        )
        resultados_aleatorios.append(metricas_aleatorias)

        f1_actual = float(metricas_aleatorias["f1_macro"])
        if f1_actual > mejor_f1:
            mejor_f1 = f1_actual
            mejor_nombre = nombre

        if temporal is not None:
            X_train_t, X_test_t, y_train_t, y_test_t, anio_prueba = temporal
            modelo_temporal = crear_pipeline(X, algoritmo)
            modelo_temporal.fit(X_train_t, y_train_t)
            resultados_temporales.append(
                evaluar_modelo(
                    nombre,
                    modo,
                    "temporal",
                    modelo_temporal,
                    X_test_t,
                    y_test_t,
                    anio_prueba,
                )
            )

    if not mejor_nombre:
        raise RuntimeError(f"No se pudo entrenar ningún modelo en modo {modo}.")

    modelo_final_modo = crear_pipeline(X, algoritmos[mejor_nombre])
    modelo_final_modo.fit(X, y)
    return {
        "modo": modo,
        "X": X,
        "columnas_excluidas": columnas_excluidas(modo),
        "mejor_nombre": mejor_nombre,
        "mejor_f1": mejor_f1,
        "modelo_final": modelo_final_modo,
        "aleatorias": resultados_aleatorios,
        "temporales": resultados_temporales,
    }


def seleccionar_modo_final(resultados: dict[str, dict]) -> tuple[str, str]:
    f1_completo = resultados["completo"]["mejor_f1"]
    f1_interpretable = resultados["interpretable"]["mejor_f1"]
    caida = f1_completo - f1_interpretable

    if (
        f1_interpretable >= F1_MINIMO_INTERPRETABLE
        and caida <= CAIDA_MAXIMA_INTERPRETABLE
    ):
        return (
            "interpretable",
            "El modo interpretable cumple el F1 mínimo y no pierde más de "
            f"{CAIDA_MAXIMA_INTERPRETABLE:.2f} frente al modo completo.",
        )

    mensaje = (
        "ADVERTENCIA: el modo interpretable no alcanzó el criterio de "
        f"aceptación (F1 >= {F1_MINIMO_INTERPRETABLE:.2f} y caída <= "
        f"{CAIDA_MAXIMA_INTERPRETABLE:.2f}); se seleccionó el modo completo."
    )
    print(mensaje)
    return "completo", mensaje


def guardar_importancia(
    modelo: Pipeline,
    path: Path,
) -> bool:
    importancias = getattr(
        modelo.named_steps["modelo"],
        "feature_importances_",
        None,
    )
    if importancias is None:
        path.unlink(missing_ok=True)
        print(
            f"ADVERTENCIA: no se pudo generar {path.name}; el modelo no expone "
            "feature_importances_."
        )
        return False

    try:
        nombres = modelo.named_steps["preprocesador"].get_feature_names_out()
        if len(nombres) != len(importancias):
            raise ValueError(
                "La cantidad de variables transformadas no coincide con las "
                "importancias."
            )
        pd.DataFrame(
            {"variable": nombres, "importancia": importancias}
        ).sort_values(
            "importancia", ascending=False
        ).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        return True
    except (AttributeError, TypeError, ValueError) as error:
        path.unlink(missing_ok=True)
        print(f"ADVERTENCIA: no se pudo generar {path.name}: {error}")
        return False


def _mejor_metrica_temporal(resultado_modo: dict) -> dict | None:
    temporales = resultado_modo["temporales"]
    if not temporales:
        return None
    nombre = resultado_modo["mejor_nombre"]
    return next(
        (metrica for metrica in temporales if metrica["modelo"] == nombre),
        None,
    )


def guardar_resultados(
    df: pd.DataFrame,
    resultados: dict[str, dict],
    modo_final: str,
    motivo_seleccion: str,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final = resultados[modo_final]
    joblib.dump(final["modelo_final"], MODEL_PATH)

    aleatorias = [
        metrica
        for resultado in resultados.values()
        for metrica in resultado["aleatorias"]
    ]
    temporales = [
        metrica
        for resultado in resultados.values()
        for metrica in resultado["temporales"]
    ]
    todas = aleatorias + temporales

    pd.DataFrame(todas).to_csv(
        METRICAS_PATH, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(aleatorias).to_csv(
        METRICAS_ALEATORIO_PATH, index=False, encoding="utf-8-sig"
    )
    if temporales:
        pd.DataFrame(temporales).to_csv(
            METRICAS_TEMPORAL_PATH, index=False, encoding="utf-8-sig"
        )
    else:
        METRICAS_TEMPORAL_PATH.unlink(missing_ok=True)

    for modo, path in (
        ("completo", METRICAS_COMPLETO_PATH),
        ("interpretable", METRICAS_INTERPRETABLE_PATH),
    ):
        pd.DataFrame(
            resultados[modo]["aleatorias"] + resultados[modo]["temporales"]
        ).to_csv(path, index=False, encoding="utf-8-sig")

    clases = {str(clase): nombre for clase, nombre in CLASES_RIESGO.items()}
    CLASES_PATH.write_text(
        json.dumps(clases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    importancia_generada = {}
    for modo, path in (
        ("completo", IMPORTANCIA_COMPLETO_PATH),
        ("interpretable", IMPORTANCIA_INTERPRETABLE_PATH),
    ):
        importancia_generada[modo] = guardar_importancia(
            resultados[modo]["modelo_final"],
            path,
        )

    if importancia_generada[modo_final]:
        importancia_final = (
            IMPORTANCIA_INTERPRETABLE_PATH
            if modo_final == "interpretable"
            else IMPORTANCIA_COMPLETO_PATH
        )
        pd.read_csv(importancia_final).to_csv(
            IMPORTANCIA_PATH,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        IMPORTANCIA_PATH.unlink(missing_ok=True)

    metrica_temporal = _mejor_metrica_temporal(final)
    anios = sorted(df["anio"].astype(int).unique().tolist())
    archivos = sorted(df["archivo_origen"].dropna().astype(str).unique().tolist())
    metadata = {
        "mejor_modelo": final["mejor_nombre"],
        "modo_entrenamiento_final": modo_final,
        "f1_macro_aleatorio": final["mejor_f1"],
        "f1_macro_temporal": (
            float(metrica_temporal["f1_macro"])
            if metrica_temporal is not None
            else None
        ),
        "anio_prueba_temporal": (
            int(metrica_temporal["anio_prueba"])
            if metrica_temporal is not None
            else None
        ),
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "numero_filas_dataset": int(df.shape[0]),
        "numero_columnas_predictoras": int(final["X"].shape[1]),
        "años_disponibles_en_dataset": anios,
        "cantidad_archivos_raw_leidos": len(archivos),
        "archivos_raw_leidos": archivos,
        "clases": clases,
        "columnas_predictoras_usadas": final["X"].columns.tolist(),
        "columnas_excluidas": final["columnas_excluidas"],
        "criterio_seleccion_modo": motivo_seleccion,
        "advertencia_metodologica": ADVERTENCIA_METODOLOGICA,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def entrenar_modelos() -> Pipeline:
    print("Cargando dataset procesado...")
    df = cargar_dataset()
    print(f"Filas: {df.shape[0]}")
    print(f"Años: {sorted(df['anio'].astype(int).unique().tolist())}")
    print(f"Archivos raw: {df['archivo_origen'].nunique()}")

    algoritmos = obtener_modelos()
    resultados = {
        modo: entrenar_modo(df, modo, algoritmos) for modo in MODOS
    }
    modo_final, motivo = seleccionar_modo_final(resultados)
    guardar_resultados(df, resultados, modo_final, motivo)

    final = resultados[modo_final]
    temporal = _mejor_metrica_temporal(final)
    print("\n" + "=" * 74)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 74)
    print(f"Modo final: {modo_final}")
    print(f"Mejor modelo: {final['mejor_nombre']}")
    print(f"F1 macro aleatorio: {final['mejor_f1']:.6f}")
    if temporal is not None:
        print(
            f"F1 macro temporal ({int(temporal['anio_prueba'])}): "
            f"{float(temporal['f1_macro']):.6f}"
        )
    print(f"Modelo guardado: {MODEL_PATH}")
    print(f"Metadata guardada: {METADATA_PATH}")
    return final["modelo_final"]


if __name__ == "__main__":
    try:
        entrenar_modelos()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error durante el entrenamiento: {error}") from error
