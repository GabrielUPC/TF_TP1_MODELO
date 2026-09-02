from calendar import monthrange
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


COLUMNAS_BASE_INDICADORES = [
    "anio",
    "mes",
    "total_ingresos",
    "total_egresos",
    "total_estancias",
    "total_pacientes_camas",
    "total_camas",
    "total_camas_disponibles",
    "total_fallecidos",
]

COLUMNAS_INDICADORES = [
    "dias_mes",
    "promedio_estancia",
    "tasa_fallecidos",
    "ratio_camas_disponibles",
    "ocupacion_estimada",
    "presion_ingresos_camas",
    "rotacion_camas",
    "diferencia_ingresos_egresos",
]


def _division_segura(
    numerador: Any,
    denominador: Any,
    valor_si_cero: Any = 0.0,
) -> Any:
    if isinstance(numerador, Real) and isinstance(denominador, Real):
        return float(numerador / denominador) if denominador != 0 else float(
            valor_si_cero
        )

    numerador_array, denominador_array = np.broadcast_arrays(
        np.asarray(numerador, dtype=float),
        np.asarray(denominador, dtype=float),
    )
    valor_cero_array = np.broadcast_to(
        np.asarray(valor_si_cero, dtype=float),
        numerador_array.shape,
    )
    resultado = valor_cero_array.copy()
    np.divide(
        numerador_array,
        denominador_array,
        out=resultado,
        where=denominador_array != 0,
    )

    serie_referencia = next(
        (
            valor
            for valor in (numerador, denominador, valor_si_cero)
            if isinstance(valor, pd.Series)
        ),
        None,
    )
    if serie_referencia is not None:
        return pd.Series(resultado, index=serie_referencia.index)
    return resultado


def dias_mes(anio: Any, mes: Any) -> Any:
    if isinstance(anio, Real) and isinstance(mes, Real):
        try:
            return monthrange(int(anio), int(mes))[1]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"No se pudo calcular los días para anio={anio}, mes={mes}."
            ) from error

    fechas = pd.to_datetime(
        {
            "year": pd.Series(anio),
            "month": pd.Series(mes),
            "day": 1,
        },
        errors="coerce",
    )
    if fechas.isna().any():
        raise ValueError(
            "Existen valores de año o mes inválidos para calcular los días del mes."
        )
    return fechas.dt.days_in_month


def capacidad_mensual(total_camas: Any, cantidad_dias: Any) -> Any:
    return total_camas * cantidad_dias


def promedio_estancia(total_estancias: Any, total_egresos: Any) -> Any:
    return _division_segura(total_estancias, total_egresos)


def tasa_fallecidos(total_fallecidos: Any, total_egresos: Any) -> Any:
    return _division_segura(total_fallecidos, total_egresos)


def ratio_camas_disponibles(
    total_camas_disponibles: Any,
    capacidad: Any,
) -> Any:
    """Consistencia: días-cama reportados / capacidad calendario teórica.

    Se conserva el nombre público. No es un porcentaje de camas libres:
    1.0 significa que los días-cama reportados coinciden con camas × días_mes.
    """
    return _division_segura(total_camas_disponibles, capacidad)


def ocupacion_estimada(
    total_pacientes_camas: Any,
    total_camas_disponibles: Any,
) -> Any:
    """Pacientes-día / días-cama disponibles reportados (ratio, sin ×100).

    Los nombres públicos se conservan por compatibilidad con D1 y la API.
    El retorno técnico 0.0 para denominador cero no acredita ocupación nula;
    esos registros requieren validación de calidad (Q06 si hay pacientes-día).
    """
    for nombre, valores in (
        ("total_pacientes_camas", total_pacientes_camas),
        ("total_camas_disponibles", total_camas_disponibles),
    ):
        try:
            numeros = np.asarray(valores, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{nombre} debe contener números finitos no negativos.") from error
        if not np.all(np.isfinite(numeros) & (numeros >= 0)):
            raise ValueError(f"{nombre} debe contener números finitos no negativos.")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            resultado = _division_segura(total_pacientes_camas, total_camas_disponibles)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("No se pudo calcular una ocupación finita.") from error
    if not np.all(np.isfinite(resultado)):
        raise ValueError("No se pudo calcular una ocupación finita.")
    return resultado


def presion_ingresos_camas(total_ingresos: Any, total_camas: Any) -> Any:
    return _division_segura(
        total_ingresos,
        total_camas,
        valor_si_cero=total_ingresos,
    )


def rotacion_camas(total_egresos: Any, total_camas: Any) -> Any:
    return _division_segura(total_egresos, total_camas)


def diferencia_ingresos_egresos(
    total_ingresos: Any,
    total_egresos: Any,
) -> Any:
    return total_ingresos - total_egresos


def _validar_campos_disponibles(campos: set[str]) -> None:
    faltantes = sorted(set(COLUMNAS_BASE_INDICADORES).difference(campos))
    if faltantes:
        raise ValueError(
            "Faltan campos base para calcular los indicadores: "
            + ", ".join(faltantes)
        )


def agregar_indicadores_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    _validar_campos_disponibles(set(df.columns))
    resultado = df.copy()

    resultado["dias_mes"] = dias_mes(resultado["anio"], resultado["mes"])
    capacidad = capacidad_mensual(
        resultado["total_camas"],
        resultado["dias_mes"],
    )
    resultado["promedio_estancia"] = promedio_estancia(
        resultado["total_estancias"],
        resultado["total_egresos"],
    )
    resultado["tasa_fallecidos"] = tasa_fallecidos(
        resultado["total_fallecidos"],
        resultado["total_egresos"],
    )
    resultado["ratio_camas_disponibles"] = ratio_camas_disponibles(
        resultado["total_camas_disponibles"],
        capacidad,
    )
    resultado["ocupacion_estimada"] = ocupacion_estimada(
        resultado["total_pacientes_camas"],
        resultado["total_camas_disponibles"],
    )
    resultado["presion_ingresos_camas"] = presion_ingresos_camas(
        resultado["total_ingresos"],
        resultado["total_camas"],
    )
    resultado["rotacion_camas"] = rotacion_camas(
        resultado["total_egresos"],
        resultado["total_camas"],
    )
    resultado["diferencia_ingresos_egresos"] = diferencia_ingresos_egresos(
        resultado["total_ingresos"],
        resultado["total_egresos"],
    )
    return resultado


def agregar_indicadores_registro(datos: Mapping[str, Any]) -> dict[str, Any]:
    _validar_campos_disponibles(set(datos))
    resultado = dict(datos)

    cantidad_dias = dias_mes(resultado["anio"], resultado["mes"])
    capacidad = capacidad_mensual(resultado["total_camas"], cantidad_dias)
    resultado.update(
        {
            "dias_mes": cantidad_dias,
            "promedio_estancia": promedio_estancia(
                resultado["total_estancias"],
                resultado["total_egresos"],
            ),
            "tasa_fallecidos": tasa_fallecidos(
                resultado["total_fallecidos"],
                resultado["total_egresos"],
            ),
            "ratio_camas_disponibles": ratio_camas_disponibles(
                resultado["total_camas_disponibles"],
                capacidad,
            ),
            "ocupacion_estimada": ocupacion_estimada(
                resultado["total_pacientes_camas"],
                resultado["total_camas_disponibles"],
            ),
            "presion_ingresos_camas": presion_ingresos_camas(
                resultado["total_ingresos"],
                resultado["total_camas"],
            ),
            "rotacion_camas": rotacion_camas(
                resultado["total_egresos"],
                resultado["total_camas"],
            ),
            "diferencia_ingresos_egresos": diferencia_ingresos_egresos(
                resultado["total_ingresos"],
                resultado["total_egresos"],
            ),
        }
    )
    return resultado
