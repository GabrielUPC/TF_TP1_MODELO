import pytest

from src.indicadores import (
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
