import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    XGBOOST_DISPONIBLE = True
except (ImportError, OSError):
    XGBClassifier = None
    XGBOOST_DISPONIBLE = False

if __package__:
    from .variables_temporales import COLUMNAS_TEMPORALES
else:
    from variables_temporales import COLUMNAS_TEMPORALES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_modelo_ipress.csv"
DATASET_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json"
)
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "modelo_ipress.joblib"
METRICAS_PATH = MODELS_DIR / "metricas_modelo.csv"
METRICAS_ALEATORIO_PATH = MODELS_DIR / "metricas_modelo_aleatorio.csv"
METRICAS_TEMPORAL_PATH = MODELS_DIR / "metricas_modelo_temporal.csv"
METRICAS_BASELINE_PATH = MODELS_DIR / "metricas_baseline.csv"
MATRIZ_TEMPORAL_PATH = MODELS_DIR / "matriz_confusion_temporal.csv"
MATRIZ_ALEATORIA_PATH = MODELS_DIR / "matriz_confusion_aleatoria.csv"
REPORTE_TEMPORAL_PATH = MODELS_DIR / "classification_report_temporal.json"
REPORTE_ALEATORIO_PATH = (
    MODELS_DIR / "classification_report_aleatorio.json"
)
CLASES_PATH = MODELS_DIR / "clases_riesgo.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
IMPORTANCIA_PATH = MODELS_DIR / "importancia_variables.csv"

VARIABLE_OBJETIVO = "nivel_riesgo_siguiente_mes_codificado"
CLASES_RIESGO = {0: "bajo", 1: "medio", 2: "alto"}
ANIO_PRUEBA_TEMPORAL: int | None = None
MARGEN_F1_BASELINE = 0.02

COLUMNAS_PREDICTORAS_BASE = [
    "anio",
    "mes",
    "departamento",
    "provincia",
    "distrito",
    "sector",
    "categoria_ipress",
    "servicio_hospitalizacion",
    "total_ingresos",
    "total_egresos",
    "total_estancias",
    "total_pacientes_camas",
    "total_camas",
    "total_camas_disponibles",
    "total_fallecidos",
    "dias_mes",
    "promedio_estancia",
    "tasa_fallecidos",
    "ratio_camas_disponibles",
    "ocupacion_estimada",
    "presion_ingresos_camas",
    "rotacion_camas",
    "diferencia_ingresos_egresos",
]
COLUMNAS_PREDICTORAS = [
    *COLUMNAS_PREDICTORAS_BASE,
    *COLUMNAS_TEMPORALES,
]

COLUMNAS_EXCLUIDAS = [
    "nivel_riesgo",
    "nivel_riesgo_codificado",
    "nivel_riesgo_actual",
    "nivel_riesgo_actual_codificado",
    "nivel_riesgo_siguiente_mes",
    "nivel_riesgo_siguiente_mes_codificado",
    "periodo_actual",
    "periodo_predicho",
    "nombre_ipress",
    "archivo_origen",
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

ADVERTENCIA_METODOLOGICA = (
    "El modelo predice el riesgo del siguiente periodo mensual usando "
    "información del periodo actual. La variable objetivo se construye "
    "mediante reglas de negocio aplicadas al periodo siguiente, por lo que el "
    "resultado debe interpretarse como una predicción operacional basada en "
    "indicadores históricos y no como validación clínica independiente. El "
    "modelo no reemplaza decisiones clínicas ni asigna camas automáticamente."
)

PAPERS_REFERENCIA = [
    {
        "referencia": "Barreto et al. (2024)",
        "aportes": (
            "Regulación de camas, limpieza de datos, comparación de modelos, "
            "métricas múltiples y control de desbalance."
        ),
    },
    {
        "referencia": "Cabral-Miranda et al. (2025)",
        "aportes": (
            "Plataforma predictiva, horizonte futuro, integración de datos, "
            "dashboard, API y despliegue escalable."
        ),
    },
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
    if df.empty:
        raise ValueError("El dataset procesado está vacío.")

    requeridas = {
        VARIABLE_OBJETIVO,
        "periodo_predicho",
        "nivel_riesgo_actual_codificado",
        *COLUMNAS_PREDICTORAS,
    }
    faltantes = sorted(requeridas.difference(df.columns))
    if faltantes:
        raise ValueError(
            "El dataset procesado no contiene columnas requeridas: "
            + ", ".join(faltantes)
        )
    return df


def cargar_metadata_dataset() -> dict[str, Any]:
    if not DATASET_METADATA_PATH.is_file():
        return {}
    return json.loads(DATASET_METADATA_PATH.read_text(encoding="utf-8"))


def separar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[COLUMNAS_PREDICTORAS].copy()
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


def _anio_objetivo(df: pd.DataFrame) -> pd.Series:
    periodos = pd.PeriodIndex(df["periodo_predicho"].astype(str), freq="M")
    return pd.Series(periodos.year, index=df.index, dtype=int)


def seleccionar_anio_prueba(
    df: pd.DataFrame,
    anio_solicitado: int | None = None,
) -> int:
    periodos = pd.PeriodIndex(df["periodo_predicho"].astype(str), freq="M")
    tabla = pd.DataFrame(
        {"anio": periodos.year, "mes": periodos.month},
        index=df.index,
    )
    meses_por_anio = tabla.groupby("anio")["mes"].agg(lambda valores: set(valores))
    anios_completos = sorted(
        int(anio)
        for anio, meses in meses_por_anio.items()
        if meses == set(range(1, 13))
    )

    if anio_solicitado is not None:
        meses = meses_por_anio.get(anio_solicitado, set())
        if not meses:
            raise ValueError(
                f"El año de prueba {anio_solicitado} no existe en los periodos "
                "objetivo."
            )
        if meses != set(range(1, 13)):
            print(
                f"ADVERTENCIA: el año de prueba {anio_solicitado} no contiene "
                "los 12 meses objetivo."
            )
        return int(anio_solicitado)

    if not anios_completos:
        raise ValueError(
            "No existe un año objetivo completo para la evaluación temporal. "
            "Use --anio-prueba para seleccionar uno explícitamente."
        )
    return anios_completos[-1]


def crear_particion_temporal(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    anio_prueba: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    anio_prueba = seleccionar_anio_prueba(df, anio_prueba)
    anios_objetivo = _anio_objetivo(df)
    mascara_train = anios_objetivo.lt(anio_prueba)
    mascara_test = anios_objetivo.eq(anio_prueba)

    X_train = X.loc[mascara_train]
    X_test = X.loc[mascara_test]
    y_train = y.loc[mascara_train]
    y_test = y.loc[mascara_test]
    if X_train.empty or X_test.empty:
        raise ValueError(
            "La partición temporal requiere datos anteriores al año de prueba "
            "y datos dentro del año de prueba."
        )
    if set(y_train.unique()) != set(CLASES_RIESGO):
        raise ValueError(
            "El entrenamiento temporal no contiene las tres clases de riesgo."
        )
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


def ajustar_pipeline(
    nombre: str,
    modelo: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    if nombre == "XGBoost":
        pesos = compute_sample_weight(class_weight="balanced", y=y_train)
        modelo.fit(X_train, y_train, modelo__sample_weight=pesos)
    else:
        modelo.fit(X_train, y_train)
    return modelo


def _specificity_macro(y_real: pd.Series, y_pred: np.ndarray) -> float:
    matriz = confusion_matrix(y_real, y_pred, labels=list(CLASES_RIESGO))
    total = matriz.sum()
    especificidades = []
    for indice in range(len(CLASES_RIESGO)):
        tp = matriz[indice, indice]
        fn = matriz[indice, :].sum() - tp
        fp = matriz[:, indice].sum() - tp
        tn = total - tp - fn - fp
        especificidades.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return float(np.mean(especificidades))


def _probabilidades_duras(y_pred: np.ndarray) -> np.ndarray:
    probabilidades = np.zeros((len(y_pred), len(CLASES_RIESGO)), dtype=float)
    probabilidades[np.arange(len(y_pred)), y_pred.astype(int)] = 1.0
    return probabilidades


def calcular_metricas(
    y_real: pd.Series,
    y_pred: np.ndarray,
    probabilidades: np.ndarray | None,
) -> dict[str, float | None]:
    probabilidades = (
        probabilidades
        if probabilidades is not None
        else _probabilidades_duras(y_pred)
    )
    try:
        auc = float(
            roc_auc_score(
                y_real,
                probabilidades,
                labels=list(CLASES_RIESGO),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = None

    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_real, y_pred)),
        "precision_macro": float(
            precision_score(y_real, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_real, y_pred, average="macro", zero_division=0)
        ),
        "specificity_macro": _specificity_macro(y_real, y_pred),
        "f1_macro": float(
            f1_score(y_real, y_pred, average="macro", zero_division=0)
        ),
        "roc_auc_ovr_macro": auc,
        "probabilidad_maxima_promedio": float(
            probabilidades.max(axis=1).mean()
        ),
    }


def _artefactos_evaluacion(
    y_real: pd.Series,
    y_pred: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    etiquetas = list(CLASES_RIESGO)
    nombres = [CLASES_RIESGO[clase] for clase in etiquetas]
    matriz = pd.DataFrame(
        confusion_matrix(y_real, y_pred, labels=etiquetas),
        index=[f"real_{nombre}" for nombre in nombres],
        columns=[f"predicho_{nombre}" for nombre in nombres],
    )
    reporte = classification_report(
        y_real,
        y_pred,
        labels=etiquetas,
        target_names=nombres,
        zero_division=0,
        output_dict=True,
    )
    return matriz, reporte


def evaluar_modelo(
    nombre: str,
    tipo_evaluacion: str,
    modelo: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    anio_prueba: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    y_pred = np.asarray(modelo.predict(X_test), dtype=int)
    probabilidades = (
        np.asarray(modelo.predict_proba(X_test))
        if hasattr(modelo, "predict_proba")
        else None
    )
    metricas = {
        "evaluacion": tipo_evaluacion,
        "modelo": nombre,
        "anio_prueba": anio_prueba,
        **calcular_metricas(y_test, y_pred, probabilidades),
    }
    matriz, reporte = _artefactos_evaluacion(y_test, y_pred)

    print("\n" + "=" * 74)
    print(f"EVALUACIÓN: {tipo_evaluacion} | MODELO: {nombre}")
    print("=" * 74)
    print("Matriz de confusión (bajo, medio, alto):")
    print(matriz.to_string())
    print(f"F1 macro: {metricas['f1_macro']:.6f}")
    print(f"ROC-AUC OVR macro: {metricas['roc_auc_ovr_macro']}")
    return metricas, matriz, reporte


def evaluar_algoritmos(
    algoritmos: dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tipo_evaluacion: str,
    anio_prueba: int | None = None,
) -> dict[str, dict[str, Any]]:
    resultados: dict[str, dict[str, Any]] = {}
    for nombre, algoritmo in algoritmos.items():
        print(f"\nEntrenando {nombre} para evaluación {tipo_evaluacion}...")
        pipeline = crear_pipeline(X_train, algoritmo)
        ajustar_pipeline(nombre, pipeline, X_train, y_train)
        metricas, matriz, reporte = evaluar_modelo(
            nombre,
            tipo_evaluacion,
            pipeline,
            X_test,
            y_test,
            anio_prueba,
        )
        resultados[nombre] = {
            "modelo": pipeline,
            "metricas": metricas,
            "matriz": matriz,
            "reporte": reporte,
        }
    return resultados


def evaluar_baselines(
    df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    indices_test: pd.Index,
    anio_prueba: int,
) -> list[dict[str, Any]]:
    clase_mayoritaria = int(y_train.value_counts().idxmax())
    predicciones = {
        "Clase_Mayoritaria": np.full(len(y_test), clase_mayoritaria, dtype=int),
        "Persistencia_Riesgo_Actual": df.loc[
            indices_test, "nivel_riesgo_actual_codificado"
        ].astype(int).to_numpy(),
        "Regla_Ocupacion_Actual": np.select(
            [
                df.loc[indices_test, "ocupacion_estimada"].ge(0.85),
                df.loc[indices_test, "ocupacion_estimada"].ge(0.70),
            ],
            [2, 1],
            default=0,
        ).astype(int),
    }
    resultados = []
    for nombre, y_pred in predicciones.items():
        resultados.append(
            {
                "evaluacion": "baseline_temporal",
                "modelo": nombre,
                "anio_prueba": anio_prueba,
                **calcular_metricas(
                    y_test,
                    y_pred,
                    _probabilidades_duras(y_pred),
                ),
            }
        )
    return resultados


def _mapear_importancias_transformadas(
    modelo: Pipeline,
    importancias: np.ndarray,
) -> pd.DataFrame:
    preprocesador = modelo.named_steps["preprocesador"]
    variables_crudas: list[str] = []
    for nombre, transformador, columnas in preprocesador.transformers_:
        if nombre == "remainder" or transformador == "drop":
            continue
        columnas = list(columnas)
        if nombre == "categoricas":
            for columna, categorias in zip(columnas, transformador.categories_):
                variables_crudas.extend([str(columna)] * len(categorias))
        else:
            variables_crudas.extend(str(columna) for columna in columnas)

    if len(variables_crudas) != len(importancias):
        raise ValueError(
            "No coincide la cantidad de variables transformadas con sus "
            "importancias."
        )
    importancia = pd.DataFrame(
        {"variable": variables_crudas, "importancia": importancias}
    )
    return (
        importancia.groupby("variable", as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def calcular_importancia_variables(
    modelo_final: Pipeline,
    modelo_temporal: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    importancias = getattr(
        modelo_final.named_steps["modelo"],
        "feature_importances_",
        None,
    )
    if importancias is not None:
        resultado = _mapear_importancias_transformadas(
            modelo_final,
            np.asarray(importancias, dtype=float),
        )
        resultado["metodo"] = "feature_importances_agregadas"
        return resultado

    muestra = X_test
    y_muestra = y_test
    if len(X_test) > 5000:
        muestra = X_test.sample(5000, random_state=42)
        y_muestra = y_test.loc[muestra.index]
    permutacion = permutation_importance(
        modelo_temporal,
        muestra,
        y_muestra,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "variable": muestra.columns,
            "importancia": permutacion.importances_mean,
            "metodo": "permutation_importance",
        }
    ).sort_values("importancia", ascending=False).reset_index(drop=True)


def _guardar_matriz(path: Path, matriz: pd.DataFrame) -> None:
    matriz.to_csv(path, index=True, index_label="clase_real", encoding="utf-8-sig")


def _serializar_json(path: Path, contenido: Any) -> None:
    path.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _limpiar_artefactos_anteriores() -> None:
    for nombre in (
        "metricas_modelo_completo.csv",
        "metricas_modelo_interpretable.csv",
        "importancia_variables_completo.csv",
        "importancia_variables_interpretable.csv",
    ):
        (MODELS_DIR / nombre).unlink(missing_ok=True)


def guardar_resultados(
    df: pd.DataFrame,
    modelo_final: Pipeline,
    mejor_nombre: str,
    resultados_temporales: dict[str, dict[str, Any]],
    resultados_aleatorios: dict[str, dict[str, Any]],
    metricas_baseline: list[dict[str, Any]],
    importancia: pd.DataFrame,
    anio_prueba: int,
    metadata_dataset: dict[str, Any],
) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _limpiar_artefactos_anteriores()
    joblib.dump(modelo_final, MODEL_PATH)

    metricas_temporales = [
        resultado["metricas"] for resultado in resultados_temporales.values()
    ]
    metricas_aleatorias = [
        resultado["metricas"] for resultado in resultados_aleatorios.values()
    ]
    pd.DataFrame(metricas_temporales + metricas_aleatorias).to_csv(
        METRICAS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metricas_temporales).to_csv(
        METRICAS_TEMPORAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metricas_aleatorias).to_csv(
        METRICAS_ALEATORIO_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metricas_baseline).to_csv(
        METRICAS_BASELINE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    mejor_temporal = resultados_temporales[mejor_nombre]
    mejor_aleatorio = resultados_aleatorios[mejor_nombre]
    _guardar_matriz(MATRIZ_TEMPORAL_PATH, mejor_temporal["matriz"])
    _guardar_matriz(MATRIZ_ALEATORIA_PATH, mejor_aleatorio["matriz"])
    _serializar_json(REPORTE_TEMPORAL_PATH, mejor_temporal["reporte"])
    _serializar_json(REPORTE_ALEATORIO_PATH, mejor_aleatorio["reporte"])
    _serializar_json(
        CLASES_PATH,
        {str(clase): nombre for clase, nombre in CLASES_RIESGO.items()},
    )
    importancia.to_csv(IMPORTANCIA_PATH, index=False, encoding="utf-8-sig")

    mejor_f1 = float(mejor_temporal["metricas"]["f1_macro"])
    mejor_f1_baseline = max(
        float(metrica["f1_macro"]) for metrica in metricas_baseline
    )
    supera_baseline = mejor_f1 > mejor_f1_baseline + MARGEN_F1_BASELINE
    if not supera_baseline:
        print(
            "ADVERTENCIA: el modelo final no supera al mejor baseline temporal "
            f"por más de {MARGEN_F1_BASELINE:.2f} de F1 macro."
        )

    distribucion = (
        df[VARIABLE_OBJETIVO]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    variables_importantes = (
        importancia.head(10)[["variable", "importancia"]].to_dict("records")
    )
    metadata = {
        "tipo_modelo": "prediccion_siguiente_mes",
        "horizonte_prediccion": "mes_siguiente",
        "variable_objetivo": VARIABLE_OBJETIVO,
        "mejor_modelo": mejor_nombre,
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "anios_disponibles_en_dataset": sorted(
            df["anio"].astype(int).unique().tolist()
        ),
        "periodos_disponibles": metadata_dataset.get(
            "periodos_disponibles",
            sorted(df["periodo_actual"].astype(str).unique().tolist()),
        ),
        "anio_prueba_temporal": int(anio_prueba),
        "numero_filas_dataset_original": int(
            metadata_dataset.get(
                "numero_filas_antes_objetivo_futuro",
                len(df),
            )
        ),
        "numero_filas_dataset_entrenamiento": int(len(df)),
        "cantidad_archivos_raw_leidos": int(
            metadata_dataset.get("cantidad_archivos_raw_leidos", 0)
        ),
        "archivos_raw_leidos": metadata_dataset.get("archivos_raw_leidos", []),
        "columnas_predictoras_usadas": COLUMNAS_PREDICTORAS,
        "columnas_excluidas": COLUMNAS_EXCLUIDAS,
        "distribucion_clases_objetivo": {
            CLASES_RIESGO[int(clase)]: int(cantidad)
            for clase, cantidad in distribucion.items()
        },
        "estrategia_desbalance": {
            "descripcion": (
                "class_weight='balanced' para Regresión Logística y Random "
                "Forest; sample_weight balanceado calculado solo sobre "
                "entrenamiento para XGBoost."
            ),
            "uso_smote": False,
            "uso_class_weight": True,
        },
        "metricas_temporales": mejor_temporal["metricas"],
        "metricas_aleatorias": mejor_aleatorio["metricas"],
        "metricas_baseline": metricas_baseline,
        "margen_f1_requerido_sobre_baseline": MARGEN_F1_BASELINE,
        "supera_baseline": supera_baseline,
        "variables_mas_importantes": variables_importantes,
        "percentiles_riesgo_actual": metadata_dataset.get(
            "percentiles_riesgo_actual",
            {},
        ),
        "metodo_percentiles": metadata_dataset.get("metodo_percentiles", ""),
        "definicion_target": metadata_dataset.get("definicion_target", {}),
        "advertencia_metodologica": ADVERTENCIA_METODOLOGICA,
        "papers_usados_como_referencia": PAPERS_REFERENCIA,
    }
    _serializar_json(METADATA_PATH, metadata)
    return metadata


def entrenar_modelos(anio_prueba: int | None = None) -> Pipeline:
    print("Cargando dataset procesado...")
    df = cargar_dataset()
    metadata_dataset = cargar_metadata_dataset()
    X, y = separar_variables(df)
    algoritmos = obtener_modelos()

    X_train_t, X_test_t, y_train_t, y_test_t, anio_prueba = (
        crear_particion_temporal(
            df,
            X,
            y,
            anio_prueba if anio_prueba is not None else ANIO_PRUEBA_TEMPORAL,
        )
    )
    print(f"Filas de entrenamiento temporal: {len(X_train_t)}")
    print(f"Filas de prueba temporal ({anio_prueba}): {len(X_test_t)}")
    resultados_temporales = evaluar_algoritmos(
        algoritmos,
        X_train_t,
        X_test_t,
        y_train_t,
        y_test_t,
        "temporal",
        anio_prueba,
    )
    mejor_nombre = max(
        resultados_temporales,
        key=lambda nombre: resultados_temporales[nombre]["metricas"]["f1_macro"],
    )

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    resultados_aleatorios = evaluar_algoritmos(
        algoritmos,
        X_train_a,
        X_test_a,
        y_train_a,
        y_test_a,
        "aleatoria_referencia",
    )
    metricas_baseline = evaluar_baselines(
        df,
        y_train_t,
        y_test_t,
        X_test_t.index,
        anio_prueba,
    )

    print(f"\nEntrenando modelo final oficial: {mejor_nombre}...")
    modelo_final = crear_pipeline(X, algoritmos[mejor_nombre])
    ajustar_pipeline(mejor_nombre, modelo_final, X, y)
    importancia = calcular_importancia_variables(
        modelo_final,
        resultados_temporales[mejor_nombre]["modelo"],
        X_test_t,
        y_test_t,
    )
    metadata = guardar_resultados(
        df,
        modelo_final,
        mejor_nombre,
        resultados_temporales,
        resultados_aleatorios,
        metricas_baseline,
        importancia,
        anio_prueba,
        metadata_dataset,
    )

    print("\n" + "=" * 74)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 74)
    print(f"Año de prueba temporal: {anio_prueba}")
    print(f"Mejor modelo: {mejor_nombre}")
    print(
        "F1 macro temporal: "
        f"{metadata['metricas_temporales']['f1_macro']:.6f}"
    )
    print(f"Supera baseline: {metadata['supera_baseline']}")
    print(f"Modelo guardado: {MODEL_PATH}")
    print(f"Metadata guardada: {METADATA_PATH}")
    return modelo_final


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de riesgo IPRESS del mes siguiente."
    )
    parser.add_argument(
        "--anio-prueba",
        type=int,
        default=None,
        help=(
            "Año objetivo para evaluación temporal. Si se omite, se usa el "
            "último año con los 12 meses disponibles."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = _parsear_argumentos()
    try:
        entrenar_modelos(argumentos.anio_prueba)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error durante el entrenamiento: {error}") from error
