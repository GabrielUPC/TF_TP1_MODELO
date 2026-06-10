import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_ipress.joblib"
CLASES_PATH = PROJECT_ROOT / "models" / "clases_riesgo.json"


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
    except (
        EOFError,
        ImportError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            f"No se pudieron cargar los artefactos del modelo: {error}"
        ) from error

    return modelo, clases


def obtener_columnas_requeridas(modelo: Any) -> list[str]:
    columnas = getattr(modelo, "feature_names_in_", None)
    if columnas is None:
        raise RuntimeError(
            "El modelo guardado no contiene el contrato de columnas de entrada."
        )
    return [str(columna) for columna in columnas]


def predecir_riesgo(datos: dict) -> dict:
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
            "nivel_riesgo": nivel,
            "nivel_riesgo_codificado": codigo,
            "probabilidad": probabilidad,
            "probabilidades_por_clase": probabilidades_por_clase,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"No se pudo realizar la predicción: {error}") from error
    except Exception as error:
        raise RuntimeError(
            f"Ocurrió un error inesperado durante la predicción: {error}"
        ) from error
