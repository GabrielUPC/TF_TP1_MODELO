import numpy as np
import pandas as pd
import pytest

from src.indicadores import (
    agregar_indicadores_dataframe,
    agregar_indicadores_registro,
    capacidad_mensual,
    ocupacion_estimada,
    ratio_camas_disponibles,
)


def test_capacidad_mensual_multiplica_camas_por_dias() -> None:
    assert capacidad_mensual(120, 31) == 3720


def test_ratio_camas_disponibles_evitar_division_entre_cero() -> None:
    assert ratio_camas_disponibles(100, 0) == 0.0


def test_ocupacion_estimada_evitar_division_entre_cero() -> None:
    assert ocupacion_estimada(50, 0) == 0.0


def test_ratio_camas_disponibles_normaliza_camas_dia() -> None:
    assert ratio_camas_disponibles(3720, 3720) == pytest.approx(1.0)


def registro(**cambios):
    return {
        "anio": 2024, "mes": 4, "total_ingresos": 30, "total_egresos": 25,
        "total_estancias": 100, "total_pacientes_camas": 240,
        "total_camas": 10, "total_camas_disponibles": 300, "total_fallecidos": 0,
    } | cambios


@pytest.mark.parametrize("mes,camas,disponibles,ocupacion,ratio", [
    (4, 10, 300, 0.8, 1.0),  # 10 camas, 30 días, 240 pacientes-día.
    (1, 10, 300, 0.8, 300/310),  # No usar 10 × 31 como denominador.
    (4, 20, 300, 0.8, 0.5),  # Cambiar camas no cambia ocupación con igual DCD.
    (4, 10, 150, 1.6, 0.5),  # No usar 300 ni recortar ocupaciones > 1.
    (4, 10, 0, 0.0, 0.0),  # Protección existente; no acredita ocupación observada.
])
def test_ocupacion_y_consistencia_en_dataframe_y_registro(mes, camas, disponibles, ocupacion, ratio):
    datos = registro(mes=mes, total_camas=camas, total_camas_disponibles=disponibles)
    original = datos.copy()
    calculado = agregar_indicadores_registro(datos)
    df = pd.DataFrame([datos], index=[17])
    original_df = df.copy(deep=True)
    calculado_df = agregar_indicadores_dataframe(df)
    for resultado in (calculado, calculado_df.loc[17].to_dict()):
        assert resultado["ocupacion_estimada"] == pytest.approx(ocupacion)
        assert resultado["ratio_camas_disponibles"] == pytest.approx(ratio)
        assert np.isfinite(resultado["ocupacion_estimada"])
        assert resultado["total_camas_disponibles"] == disponibles
        assert resultado["total_pacientes_camas"] == 240
    assert calculado_df.index.tolist() == [17]
    assert datos == original
    pd.testing.assert_frame_equal(df, original_df)


def test_dataframe_mixto_con_cero_no_genera_infinito():
    df = pd.DataFrame([registro(), registro(total_camas_disponibles=0)], index=[4, 9])
    assert agregar_indicadores_dataframe(df).ocupacion_estimada.tolist() == [0.8, 0.0]


@pytest.mark.parametrize("pacientes,disponibles", [
    (-1, 300), (240, -1), (float("nan"), 300), (240, float("inf")),
    ("inválido", 300), (240, None),
])
@pytest.mark.parametrize("vectorial", [False, True])
def test_ocupacion_rechaza_valores_invalidos(pacientes, disponibles, vectorial):
    if vectorial:
        pacientes, disponibles = pd.Series([pacientes]), pd.Series([disponibles])
    with pytest.raises(ValueError, match="números finitos no negativos"):
        ocupacion_estimada(pacientes, disponibles)


def test_ocupacion_rechaza_desbordamiento():
    with pytest.raises(ValueError, match="ocupación finita"):
        ocupacion_estimada(1e308, 1e-308)


def test_preparacion_usa_dias_cama_reportados_sin_cambiar_nombres():
    from src.preparar_dataset import crear_indicadores, RENOMBRE_COLUMNAS

    inverso = {destino: origen for origen, destino in RENOMBRE_COLUMNAS.items()}
    raw = pd.DataFrame([{inverso.get(k, k): v for k, v in registro(total_camas_disponibles=150).items()}])
    resultado = crear_indicadores(raw).iloc[0]
    assert resultado.ocupacion_estimada == pytest.approx(1.6)
    assert resultado.ratio_camas_disponibles == pytest.approx(0.5)
    assert resultado.total_camas_disponibles == 150
