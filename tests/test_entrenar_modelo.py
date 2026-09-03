import json

import numpy as np
import pandas as pd

from src.entrenar_modelo import (
    COLUMNAS_EXCLUIDAS,
    COLUMNAS_PREDICTORAS,
    METADATA_PATH,
    MARGEN_F1_BASELINE,
    evaluar_baselines,
    seleccionar_anio_prueba,
)


def test_no_se_usan_etiquetas_actuales_ni_identificadores_como_predictoras() -> None:
    prohibidas = {
        "nivel_riesgo_actual",
        "nivel_riesgo_actual_codificado",
        "nivel_riesgo_siguiente_mes",
        "nivel_riesgo_siguiente_mes_codificado",
        "nombre_ipress",
        "archivo_origen",
        "codigo_ipress",
        "ubigeo",
        "id_hospitalizacion",
    }

    assert prohibidas.isdisjoint(COLUMNAS_PREDICTORAS)
    assert prohibidas.issubset(COLUMNAS_EXCLUIDAS)


def test_selecciona_ultimo_anio_objetivo_completo() -> None:
    periodos = [
        *(f"2024-{mes:02d}" for mes in range(1, 13)),
        *(f"2025-{mes:02d}" for mes in range(1, 5)),
    ]
    df = pd.DataFrame({"periodo_predicho": periodos})

    assert seleccionar_anio_prueba(df) == 2024


def test_evaluar_baselines_genera_tres_referencias() -> None:
    indices = pd.Index([3, 4, 5])
    df = pd.DataFrame(
        {
            "nivel_riesgo_actual_codificado": [0, 1, 2, 0, 1, 2],
            "ocupacion_estimada": [0.5, 0.75, 0.9, 0.4, 0.8, 0.95],
        }
    )
    y_train = pd.Series([0, 0, 1], index=[0, 1, 2])
    y_test = pd.Series([0, 1, 2], index=indices)

    metricas = evaluar_baselines(df, y_train, y_test, indices, 2025)

    assert {metrica["modelo"] for metrica in metricas} == {
        "Clase_Mayoritaria",
        "Persistencia_Riesgo_Actual",
        "Regla_Ocupacion_Actual",
    }
    assert all(np.isfinite(metrica["f1_macro"]) for metrica in metricas)


def test_metadata_guardada_contiene_trazabilidad_del_modelo() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["tipo_modelo"] == "prediccion_siguiente_mes"
    assert metadata["variable_objetivo"] == (
        "nivel_riesgo_siguiente_mes_codificado"
    )
    assert isinstance(metadata["supera_baseline"], bool)
    if metadata.get("es_modelo_final_produccion"):
        comparacion = metadata["comparacion_baselines_vigente"]
        assert comparacion["motivo"]
        assert comparacion["anios"] == [2018, 2021, 2022, 2023, 2024]
        assert comparacion["margen_f1_requerido"] == MARGEN_F1_BASELINE
        if comparacion["evidencia_verificada"]:
            esperado = (
                comparacion["f1_macro_regla_final_promedio"]
                > comparacion["mejor_f1_macro_baseline_promedio"]
                + comparacion["margen_f1_requerido"]
            )
            assert metadata["supera_baseline"] is esperado
        else:
            assert metadata["supera_baseline"] is False
