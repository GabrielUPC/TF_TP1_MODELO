import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.interpretacion import (
    factores_explicativos_riesgo,
    interpretar_semaforo,
)
from src.soporte_decision import generar_soporte_decision
from src.modelo_final import (
    REGLA_FINAL, SIGNIFICADO_PROBABILIDAD, SIGNIFICADO_INDICE,
    probabilidades_ordenadas, decidir_clases,
)
from src.tratamiento_capacidad import sha256_archivo


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
            "Ejecute primero: python -m src.entrenar_modelo_final"
        )
    if not CLASES_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de clases en {CLASES_PATH}. "
            "Ejecute primero: python -m src.entrenar_modelo_final"
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
    metadata = cargar_metadata()
    if (metadata.get("es_modelo_final_produccion") is not True
            or metadata.get("regla_decision") != REGLA_FINAL
            or getattr(modelo, "regla_decision_", None) != REGLA_FINAL
            or metadata.get("algoritmo_final") != "XGBoost"
            or metadata.get("conjunto_features") != "D"
            or metadata.get("modelo_sha256") != sha256_archivo(MODEL_PATH)
            or metadata.get("lista_features") != obtener_columnas_requeridas(modelo)):
        raise RuntimeError(
            "Modelo y metadata no corresponden al contrato final XGBoost D. "
            "Ejecute python -m src.entrenar_modelo_final y reinicie la API."
        )
    return modelo, clases


@lru_cache(maxsize=1)
def cargar_metadata() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró la metadata en {METADATA_PATH}. "
            "Ejecute primero: python -m src.entrenar_modelo_final"
        )
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"No se pudo cargar la metadata: {error}") from error


@lru_cache(maxsize=1)
def cargar_importancia() -> pd.DataFrame:
    metadata = cargar_metadata()
    if metadata.get("es_modelo_final_produccion"):
        # No presentar la importancia del CSV productivo antiguo como vigente.
        return pd.DataFrame(metadata.get("importancia_variables_final", []),
                            columns=["variable", "importancia"])
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

def calcular_riesgo_insuficiencia_capacidad(
    nivel: str,
    confianza: float,
) -> float:
    """Índice visual legado, NO probabilidad calibrada; conservado por contrato API.

    No interviene en decidir la clase ni en las métricas del modelo.
    """
    confianza = max(0.0, min(float(confianza), 1.0))
    nivel_normalizado = nivel.lower()

    if nivel_normalizado == "bajo":
        return max(0.0, min(0.33, (1.0 - confianza) * 0.33))

    if nivel_normalizado == "medio":
        return max(0.33, min(0.66, 0.33 + confianza * 0.33))

    if nivel_normalizado == "alto":
        return max(0.66, min(1.0, 0.66 + confianza * 0.34))

    return 0.0

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
        probabilidades = probabilidades_ordenadas(modelo, entrada)
        codigo = int(decidir_clases(probabilidades)[0])
        nivel = clases.get(str(codigo))
        if nivel is None:
            raise ValueError(
                f"La clase predicha '{codigo}' no existe en el archivo de clases."
            )

        probabilidades_por_clase = {
            clases[str(c)]: float(probabilidades[0, c]) for c in (0, 1, 2)
        }
        # La regla puede apartarse de argmax: devolver p de la clase FINAL.
        probabilidad = float(probabilidades[0, codigo])

        semaforo = interpretar_semaforo(nivel)
        soporte = generar_soporte_decision(datos, nivel, probabilidad)

        return {
            "nivel_riesgo_predicho": nivel,
            "nivel_riesgo_codificado": codigo,
            "probabilidad": probabilidad,
            "riesgo_insuficiencia_capacidad": calcular_riesgo_insuficiencia_capacidad(
                nivel,
                probabilidad,
            ),
            "probabilidades_por_clase": probabilidades_por_clase,
            "variables_principales": obtener_variables_principales(datos),
            "color_semaforo": semaforo["color_semaforo"],
            "interpretacion_riesgo": semaforo["interpretacion_riesgo"],
            "recomendacion_riesgo": semaforo["recomendacion_riesgo"],
            "factores_explicativos": factores_explicativos_riesgo(datos, nivel),
            "probabilidad_riesgo_bajo": float(
                probabilidades_por_clase.get("bajo", 0.0)
            ),
            "probabilidad_riesgo_medio": float(
                probabilidades_por_clase.get("medio", 0.0)
            ),
            "probabilidad_riesgo_alto": float(
                probabilidades_por_clase.get("alto", 0.0)
            ),
            **soporte,
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
        "algoritmo_final": metadata.get("algoritmo_final"),
        "conjunto_features": metadata.get("conjunto_features"),
        "numero_features": metadata.get("numero_features"),
        "regla_decision": metadata.get("regla_decision"),
        "es_modelo_final_produccion": metadata.get("es_modelo_final_produccion", False),
        "definicion_probabilidad": SIGNIFICADO_PROBABILIDAD,
        "definicion_riesgo_insuficiencia_capacidad": SIGNIFICADO_INDICE,
        "resultados_desarrollo": metadata.get("resultados_desarrollo"),
        "resultados_comprobacion_2025": metadata.get("resultados_comprobacion_2025"),
        "advertencia_2025": metadata.get("advertencia_2025"),
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
