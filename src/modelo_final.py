"""Contrato congelado XGBoost D; sin entrenamiento ni selección al importar."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.variables_temporales import agregar_variables_temporales, preparar_registro_con_historial
from src.variables_temporales_experimentales import agregar_features_candidatas

REGLA_FINAL = {
    "nombre": "combinada_0.35_0.20", "tipo": "combinada",
    "umbral_alto": 0.35, "umbral_proteccion_bajo": 0.20,
}
ANIOS_DESARROLLO = [2018, 2021, 2022, 2023, 2024]
SIGNIFICADO_PROBABILIDAD = "Probabilidad sin recalibrar de la clase FINAL predicha; no max(p)."
SIGNIFICADO_INDICE = (
    "Índice visual/operativo derivado de nivel y probabilidad de clase final. "
    "No es una probabilidad calibrada de insuficiencia; no decide la clase ni evalúa desempeño."
)


def probabilidades_ordenadas(modelo, entrada):
    """Valida classes_ explícitamente; retorna columnas bajo/medio/alto sin alterar p."""
    clases = np.asarray(getattr(modelo, "classes_", []))
    if clases.shape != (3,) or set(clases.tolist()) != {0, 1, 2}:
        raise ValueError("classes_ debe contener exactamente las clases numéricas 0, 1 y 2.")
    if not callable(getattr(modelo, "predict_proba", None)):
        raise ValueError("El modelo final requiere predict_proba; no se admite fallback a predict.")
    p = np.asarray(modelo.predict_proba(entrada), dtype=float)
    if p.shape != (len(entrada), 3):
        raise ValueError("predict_proba debe devolver una matriz de tres clases por registro.")
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Probabilidades inválidas: deben ser finitas, de 0 a 1 y sumar 1.")
    return p[:, [int(np.flatnonzero(clases == c)[0]) for c in (0, 1, 2)]]


def decidir_clases(p):
    """Recibe probabilidades ya validadas y ordenadas; prioridad Alto, protección, argmax."""
    pred = np.argmax(p, axis=1)
    return np.where(p[:, 2] >= REGLA_FINAL["umbral_alto"], 2,
        np.where((pred == 0) & (p[:, 2] >= REGLA_FINAL["umbral_proteccion_bajo"]), 1, pred))


class ModeloFinalD:
    """Pipeline ajustado con contrato estricto de las 69 columnas D en su orden original."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.feature_names_in_ = np.asarray(pipeline.feature_names_in_)
        self.classes_ = np.array([0, 1, 2])
        self.regla_decision_ = dict(REGLA_FINAL)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Se requieren exactamente las columnas D y su orden de entrenamiento.")
        return probabilidades_ordenadas(self.pipeline, X)

    def predict(self, X):
        return decidir_clases(self.predict_proba(X))


def preparar_entrada_d(actual, historial):
    """Mismas fórmulas D del experimento; no imputar ventanas incompletas ni usar t+1.

    Entradas con indicadores ya calculados. La validación de grupo/fecha conserva
    el contrato existente. El historial aportado debe contener meses validados.
    """
    from src.preparar_dataset import crear_riesgo_actual

    preparar_registro_con_historial(actual, historial)  # valida grupo, duplicados y futuro
    df = agregar_variables_temporales(pd.DataFrame([*historial, actual]))
    df, _ = crear_riesgo_actual(df)
    df = agregar_features_candidatas(df)
    fila = df.iloc[[-1]].reset_index(drop=True)
    periodo = pd.Period(f"{int(actual['anio']):04d}-{int(actual['mes']):02d}", freq="M")
    previos = {pd.Period(f"{int(r['anio']):04d}-{int(r['mes']):02d}", freq="M") for r in historial}
    completo = {periodo-k for k in range(1, 6)}.issubset(previos)
    # Una racha que alcanza el inicio del historial podría estar truncada.
    # No cambiamos su fórmula: avisamos y permitimos enviar más de doce meses.
    if int(fila.iloc[0].meses_consecutivos_alto) == len(df):
        completo = False
    return fila, completo
