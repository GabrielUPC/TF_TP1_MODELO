"""Contrato reutilizable de inferencia para el candidato binario versionado."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features_binarias import COLUMNAS_PREDICTORAS_BINARIAS, validar_predictoras

VERSION_MODELO = "modelo_ipress_binario_v1"
TARGET = "alta_presion_capacidad_siguiente_mes"
THRESHOLD_OPERACIONAL = 0.36
SIGNIFICADO_PROBABILIDAD = (
    "Probabilidad estimada de que la ocupación del servicio supere 90% "
    "en el siguiente mes calendario exacto."
)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "modelo_ipress_binario_v1.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata_binario_v1.json"


def sha256_archivo(path):
    with Path(path).open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def validar_entrada(entrada, features=None, *, permitir_columnas_adicionales=False):
    """Valida nombres y devuelve las features en el orden contractual."""
    if not isinstance(entrada, pd.DataFrame) or entrada.empty:
        raise ValueError("La entrada debe ser un DataFrame no vacío.")
    esperadas = validar_predictoras(features or COLUMNAS_PREDICTORAS_BINARIAS)
    faltantes = [c for c in esperadas if c not in entrada.columns]
    adicionales = [c for c in entrada.columns if c not in esperadas]
    if faltantes:
        raise ValueError("Faltan features requeridas: " + ", ".join(faltantes))
    if adicionales and not permitir_columnas_adicionales:
        raise ValueError("Columnas adicionales no permitidas: " + ", ".join(adicionales))
    X = entrada.loc[:, esperadas].copy()
    for columna in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[columna] = X[columna].astype(object).where(X[columna].notna(), np.nan)
    numericas = X.select_dtypes(include="number")
    if np.isinf(numericas.to_numpy(dtype=float)).any():
        raise ValueError("La entrada contiene infinitos.")
    return X


def validar_artefacto(artefacto):
    if not isinstance(artefacto, dict):
        raise ValueError("Artefacto binario inválido.")
    requeridas = {"pipeline", "version_modelo", "target", "threshold_operacional", "features"}
    if not requeridas.issubset(artefacto):
        raise ValueError("Artefacto binario incompleto.")
    if (artefacto["version_modelo"] != VERSION_MODELO
            or artefacto["target"] != TARGET
            or artefacto["threshold_operacional"] != THRESHOLD_OPERACIONAL
            or artefacto["features"] != COLUMNAS_PREDICTORAS_BINARIAS):
        raise ValueError("El artefacto no coincide con el contrato binario v1.")
    if not callable(getattr(artefacto["pipeline"], "predict_proba", None)):
        raise ValueError("El pipeline no implementa predict_proba.")
    orden_pipeline = getattr(artefacto["pipeline"], "feature_names_in_", None)
    if orden_pipeline is not None and list(orden_pipeline) != artefacto["features"]:
        raise ValueError("El orden de features del pipeline no coincide con el artefacto.")
    return artefacto


def cargar_modelo_binario(modelo_path, metadata_path):
    """Carga candidato y metadata, y comprueba su SHA y contrato."""
    modelo_path, metadata_path = Path(modelo_path), Path(metadata_path)
    if not modelo_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("No existe el modelo binario o su metadata.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("modelo_sha256") != sha256_archivo(modelo_path):
        raise ValueError("El SHA-256 del modelo no coincide con la metadata.")
    artefacto = validar_artefacto(joblib.load(modelo_path))
    if (metadata.get("version_modelo") != artefacto["version_modelo"]
            or metadata.get("target") != TARGET
            or metadata.get("modelo_sha256") != sha256_archivo(modelo_path)
            or metadata.get("threshold_operacional") != artefacto["threshold_operacional"]
            or metadata.get("lista_features") != artefacto["features"]
            or metadata.get("numero_features") != len(artefacto["features"])):
        raise ValueError("Metadata y artefacto tienen contratos diferentes.")
    return artefacto, metadata


@lru_cache(maxsize=1)
def cargar_artefactos_binarios():
    """Carga perezosa y cacheada del candidato binario con su metadata."""
    return cargar_modelo_binario(MODEL_PATH, METADATA_PATH)


def limpiar_cache_binaria():
    cargar_artefactos_binarios.cache_clear()


def predecir_alta_presion(artefacto, entrada, *, permitir_columnas_adicionales=False):
    """Devuelve P(Y=1), threshold fijo y alerta; no crea niveles Bajo/Medio/Alto."""
    artefacto = validar_artefacto(artefacto)
    X = validar_entrada(
        entrada, artefacto["features"],
        permitir_columnas_adicionales=permitir_columnas_adicionales,
    )
    pipeline = artefacto["pipeline"]
    probabilidades = np.asarray(pipeline.predict_proba(X), dtype=float)
    clases = list(pipeline.classes_)
    if (set(clases) != {0, 1} or probabilidades.shape != (len(X), 2)
            or not np.isfinite(probabilidades).all()
            or (probabilidades < 0).any() or (probabilidades > 1).any()
            or not np.allclose(probabilidades.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("predict_proba no devolvió probabilidades binarias válidas.")
    score = probabilidades[:, clases.index(1)]
    threshold = artefacto["threshold_operacional"]
    return pd.DataFrame({
        "probabilidad_alta_presion": score,
        "threshold": np.full(len(score), threshold),
        "alerta_alta_presion": score >= threshold,
    }, index=entrada.index)
