from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


COLUMNAS_GRUPO_TEMPORAL = [
    "codigo_ipress",
    "servicio_hospitalizacion",
]

COLUMNAS_PROMEDIO_MOVIL = {
    "ocupacion_estimada": "promedio_movil_3m_ocupacion",
    "presion_ingresos_camas": "promedio_movil_3m_presion",
    "total_ingresos": "promedio_movil_3m_ingresos",
    "total_egresos": "promedio_movil_3m_egresos",
    "total_estancias": "promedio_movil_3m_estancias",
}

COLUMNAS_TENDENCIA = {
    "ocupacion_estimada": "tendencia_ocupacion_1m",
    "presion_ingresos_camas": "tendencia_presion_1m",
    "total_ingresos": "tendencia_ingresos_1m",
    "total_egresos": "tendencia_egresos_1m",
}

COLUMNAS_TEMPORALES = [
    "trimestre",
    "semestre",
    "mes_sin",
    "mes_cos",
    "es_fin_de_anio",
    *COLUMNAS_PROMEDIO_MOVIL.values(),
    *COLUMNAS_TENDENCIA.values(),
]


def _validar_columnas(df: pd.DataFrame, columnas: Sequence[str]) -> None:
    faltantes = sorted(set(columnas).difference(df.columns))
    if faltantes:
        raise ValueError(
            "Faltan columnas para calcular variables temporales: "
            + ", ".join(faltantes)
        )


def _ordinal_mensual(anio: pd.Series, mes: pd.Series) -> pd.Series:
    return anio.astype(int) * 12 + mes.astype(int) - 1


def periodo_siguiente(anio: int, mes: int) -> str:
    periodo = pd.Period(f"{int(anio):04d}-{int(mes):02d}", freq="M") + 1
    return str(periodo)


def agregar_variables_temporales(df: pd.DataFrame) -> pd.DataFrame:
    columnas_requeridas = [
        *COLUMNAS_GRUPO_TEMPORAL,
        "anio",
        "mes",
        *COLUMNAS_PROMEDIO_MOVIL,
        *COLUMNAS_TENDENCIA,
    ]
    _validar_columnas(df, columnas_requeridas)

    resultado = df.copy()
    resultado["_orden_original"] = np.arange(len(resultado))
    resultado["_ordinal_mes"] = _ordinal_mensual(
        resultado["anio"],
        resultado["mes"],
    )
    resultado = resultado.sort_values(
        [*COLUMNAS_GRUPO_TEMPORAL, "_ordinal_mes", "_orden_original"]
    )

    duplicados = resultado.duplicated(
        [*COLUMNAS_GRUPO_TEMPORAL, "_ordinal_mes"],
        keep=False,
    )
    if duplicados.any():
        raise ValueError(
            "Existen registros duplicados para una misma IPRESS, servicio y "
            "periodo mensual; no se pueden calcular tendencias confiables."
        )

    resultado["trimestre"] = ((resultado["mes"].astype(int) - 1) // 3 + 1)
    resultado["semestre"] = ((resultado["mes"].astype(int) - 1) // 6 + 1)
    angulo = 2 * np.pi * resultado["mes"].astype(float) / 12
    resultado["mes_sin"] = np.sin(angulo)
    resultado["mes_cos"] = np.cos(angulo)
    resultado["es_fin_de_anio"] = resultado["mes"].astype(int).eq(12).astype(int)

    grupos = resultado.groupby(
        COLUMNAS_GRUPO_TEMPORAL,
        sort=False,
        observed=True,
    )
    ordinal_anterior = grupos["_ordinal_mes"].shift(1)
    ordinal_hace_dos = grupos["_ordinal_mes"].shift(2)
    mes_anterior_contiguo = resultado["_ordinal_mes"].sub(ordinal_anterior).eq(1)
    dos_meses_contiguos = (
        resultado["_ordinal_mes"].sub(ordinal_hace_dos).eq(2)
        & ordinal_anterior.sub(ordinal_hace_dos).eq(1)
    )

    for columna, salida in COLUMNAS_PROMEDIO_MOVIL.items():
        anterior = grupos[columna].shift(1).where(mes_anterior_contiguo)
        hace_dos = grupos[columna].shift(2).where(dos_meses_contiguos)
        resultado[salida] = pd.concat(
            [resultado[columna], anterior, hace_dos],
            axis=1,
        ).mean(axis=1)

    for columna, salida in COLUMNAS_TENDENCIA.items():
        anterior = grupos[columna].shift(1)
        resultado[salida] = (
            resultado[columna].sub(anterior).where(mes_anterior_contiguo, 0.0)
        )

    return (
        resultado.sort_values("_orden_original")
        .drop(columns=["_orden_original", "_ordinal_mes"])
        .reset_index(drop=True)
    )


def preparar_registro_con_historial(
    registro_actual: dict[str, Any],
    historial: Sequence[dict[str, Any]],
) -> tuple[pd.DataFrame, bool]:
    filas = [*historial, registro_actual]
    df = pd.DataFrame(filas)
    _validar_columnas(
        df,
        [*COLUMNAS_GRUPO_TEMPORAL, "anio", "mes"],
    )

    grupo_actual = tuple(
        str(registro_actual[columna]) for columna in COLUMNAS_GRUPO_TEMPORAL
    )
    grupos = {
        tuple(str(fila[columna]) for columna in COLUMNAS_GRUPO_TEMPORAL)
        for fila in filas
    }
    if grupos != {grupo_actual}:
        raise ValueError(
            "El historial debe corresponder a la misma IPRESS y servicio "
            "hospitalario que el registro actual."
        )

    periodo_actual = pd.Period(
        f"{int(registro_actual['anio']):04d}-"
        f"{int(registro_actual['mes']):02d}",
        freq="M",
    )
    periodos_historial = [
        pd.Period(
            f"{int(fila['anio']):04d}-{int(fila['mes']):02d}",
            freq="M",
        )
        for fila in historial
    ]
    if any(periodo >= periodo_actual for periodo in periodos_historial):
        raise ValueError(
            "Todos los registros históricos deben ser anteriores al periodo "
            "actual."
        )
    if len(set(periodos_historial)) != len(periodos_historial):
        raise ValueError("El historial contiene periodos mensuales duplicados.")

    resultado = agregar_variables_temporales(df)
    mascara_actual = (
        resultado["anio"].astype(int).eq(periodo_actual.year)
        & resultado["mes"].astype(int).eq(periodo_actual.month)
    )
    fila_actual = resultado.loc[mascara_actual].tail(1)

    periodos_requeridos = {periodo_actual - 1, periodo_actual - 2}
    historial_completo = periodos_requeridos.issubset(set(periodos_historial))
    return fila_actual.reset_index(drop=True), historial_completo
