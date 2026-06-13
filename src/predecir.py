import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_ipress.joblib"
CLASES_PATH = PROJECT_ROOT / "models" / "clases_riesgo.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
IMPORTANCIA_PATH = PROJECT_ROOT / "models" / "importancia_variables.csv"
CLASES_ESPERADAS = {"0": "bajo", "1": "medio", "2": "alto"}


@lru_cache(maxsize=1)
def cargar_artefactos() -> tuple[Any, dict[str, str]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecute primero: py src/entrenar_modelo.py"
        )
    if not CLASES_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de clases en {CLASES_PATH}. "
            "Ejecute primero: py src/entrenar_modelo.py"
        )
    try:
        modelo = joblib.load(MODEL_PATH)
        clases = json.loads(CLASES_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(
            f"No se pudieron cargar los artefactos del modelo: {error}"
        ) from error

    if clases != CLASES_ESPERADAS:
        raise RuntimeError(
            "El archivo de clases no contiene el mapeo esperado: "
            '{"0": "bajo", "1": "medio", "2": "alto"}.'
        )
    return modelo, clases


@lru_cache(maxsize=1)
def cargar_metadata() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró la metadata en {METADATA_PATH}. "
            "Ejecute primero: py src/entrenar_modelo.py"
        )
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"No se pudo cargar la metadata: {error}") from error


@lru_cache(maxsize=1)
def cargar_importancia() -> pd.DataFrame:
    if not IMPORTANCIA_PATH.is_file():
        return pd.DataFrame(columns=["variable", "importancia"])
    try:
        return pd.read_csv(IMPORTANCIA_PATH).sort_values(
            "importancia",
            ascending=False,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise RuntimeError(
            f"No se pudo cargar la importancia de variables: {error}"
        ) from error


def limpiar_caches() -> None:
    cargar_artefactos.cache_clear()
    cargar_metadata.cache_clear()
    cargar_importancia.cache_clear()


def obtener_columnas_requeridas(modelo: Any) -> list[str]:
    columnas = getattr(modelo, "feature_names_in_", None)
    if columnas is None:
        raise RuntimeError(
            "El modelo guardado no contiene el contrato de columnas de entrada."
        )
    return [str(columna) for columna in columnas]


def _valor_serializable(valor: Any) -> Any:
    if isinstance(valor, np.generic):
        return valor.item()
    if pd.isna(valor):
        return None
    return valor


def obtener_variables_principales(
    datos: dict[str, Any],
    limite: int = 5,
) -> list[dict[str, Any]]:
    importancia = cargar_importancia()
    principales = []
    for variable in importancia["variable"].astype(str):
        if variable not in datos:
            continue
        principales.append(
            {
                "variable": variable,
                "valor": _valor_serializable(datos[variable]),
            }
        )
        if len(principales) == limite:
            break
    return principales


def predecir_riesgo(datos: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(datos, dict):
        raise TypeError("Los datos de entrada deben enviarse como un diccionario.")

    modelo, clases = cargar_artefactos()
    columnas_requeridas = obtener_columnas_requeridas(modelo)
    faltantes = [
        columna for columna in columnas_requeridas if columna not in datos
    ]
    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas para la predicción: "
            + ", ".join(faltantes)
        )

    entrada = pd.DataFrame(
        [{columna: datos[columna] for columna in columnas_requeridas}]
    )
    try:
        codigo = int(modelo.predict(entrada)[0])
        nivel = clases.get(str(codigo))
        if nivel is None:
            raise ValueError(
                f"La clase predicha '{codigo}' no existe en el archivo de clases."
            )

        probabilidades_por_clase = {
            nombre: 0.0 for nombre in clases.values()
        }
        probabilidad = 0.0
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(entrada)[0]
            clases_modelo = getattr(modelo, "classes_", range(len(probabilidades)))
            for clase, valor in zip(clases_modelo, probabilidades):
                nombre_clase = clases.get(str(int(clase)), str(clase))
                probabilidades_por_clase[nombre_clase] = float(valor)
            probabilidad = float(max(probabilidades))

        return {
            "nivel_riesgo_predicho": nivel,
            "nivel_riesgo_codificado": codigo,
            "probabilidad": probabilidad,
            "probabilidades_por_clase": probabilidades_por_clase,
            "variables_principales": obtener_variables_principales(datos),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"No se pudo realizar la predicción: {error}") from error
    except Exception as error:
        raise RuntimeError(
            f"Ocurrió un error inesperado durante la predicción: {error}"
        ) from error


def obtener_metadata_publica() -> dict[str, Any]:
    metadata = cargar_metadata()
    metricas_temporales = metadata.get("metricas_temporales", {})
    return {
        "tipo_modelo": metadata.get("tipo_modelo"),
        "horizonte_prediccion": metadata.get("horizonte_prediccion"),
        "mejor_modelo": metadata.get("mejor_modelo"),
        "fecha_entrenamiento": metadata.get("fecha_entrenamiento"),
        "anios_disponibles_en_dataset": metadata.get(
            "anios_disponibles_en_dataset",
            [],
        ),
        "anio_prueba_temporal": metadata.get("anio_prueba_temporal"),
        "variable_objetivo": metadata.get("variable_objetivo"),
        "f1_macro_temporal": metricas_temporales.get("f1_macro"),
        "columnas_predictoras_usadas": metadata.get(
            "columnas_predictoras_usadas",
            [],
        ),
        "variables_mas_importantes": metadata.get(
            "variables_mas_importantes",
            [],
        ),
        "estrategia_desbalance": metadata.get("estrategia_desbalance"),
        "supera_baseline": metadata.get("supera_baseline"),
        "advertencia_metodologica": metadata.get(
            "advertencia_metodologica",
        ),
    }
