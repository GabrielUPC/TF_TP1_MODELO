import pandas as pd

from src.validar_plantilla import MENSAJE_DUPLICADO, validar


def fila_valida(**overrides: object) -> dict[str, object]:
    fila = {
        "codigo_renipress": "00006207",
        "anio": "2026",
        "mes": "3",
        "servicio_hospitalario": "Medicina",
        "ingresos": "80",
        "egresos": "75",
        "estancias": "350",
        "pacientes_cama": "704",
        "camas_totales": "100",
        "camas_disponibles_habilitadas": "715",
    }
    fila.update(overrides)
    return fila


def test_validar_plantilla_acepta_alias_y_normaliza_columnas() -> None:
    resultado = validar(pd.DataFrame([fila_valida()]))

    assert resultado["errores"] == []
    assert resultado["registros_validos"] == 1
    assert "codigo_ipress" in resultado["dataframe"].columns
    assert "total_camas_disponibles" in resultado["dataframe"].columns


def test_validar_plantilla_rechaza_duplicados_por_granularidad() -> None:
    resultado = validar(pd.DataFrame([fila_valida(), fila_valida()]))

    assert MENSAJE_DUPLICADO in resultado["errores"]
    assert resultado["registros_validos"] == 0
    assert len(resultado["duplicados"]) == 2
