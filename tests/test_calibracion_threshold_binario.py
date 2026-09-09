import json

import pandas as pd
import pytest

from src import analizar_calibracion_threshold as act


def probabilidades():
    filas = []
    for anio in act.ANIOS_DESARROLLO:
        for indice, (y, p) in enumerate([(0, .05), (0, .30), (1, .55), (1, .90)]):
            for modelo, delta in zip(act.MODELOS, [0, .01, -.01]):
                filas.append({
                    "indice_dataset": anio * 10 + indice, "modelo": modelo,
                    "tipo_score": "probabilidad_sin_calibracion",
                    "anio_prueba": anio,
                    "familia_servicio": ["hospitalizacion", "cuidados_criticos"][indice % 2],
                    "y_real": y, "score_positivo": min(max(p + delta, 0), 1),
                    "codigo_ipress": "00000001", "servicio_hospitalizacion": "SERVICIO",
                    "periodo_actual": f"{anio-1}-12", "periodo_predicho": f"{anio}-01",
                })
    return pd.DataFrame(filas)


def test_metricas_threshold_incluyen_costos_y_npv():
    m = act.metricas_threshold([0, 0, 1, 1], [.1, .6, .4, .9], .5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["recall"] == m["especificidad"] == m["precision_ppv"] == .5
    assert m["npv"] == m["f1"] == m["balanced_accuracy"] == .5
    assert m["tasa_alertas"] == m["tasa_falsos_negativos"] == .5
    assert m["brier_score"] == pytest.approx((.1**2 + .6**2 + .6**2 + .1**2) / 4)


def test_calibracion_diez_bins_brier_y_extremo_uno():
    df = pd.DataFrame({"modelo": "XGBoost", "y_real": [0, 1, 1],
                       "score_positivo": [0, .55, 1]})
    tabla = act.tabla_calibracion(df, ("modelo",))
    assert len(tabla) == 10 and tabla.n.sum() == 3
    assert tabla.loc[tabla.bin.eq(9), "n"].item() == 1
    assert tabla.brier_grupo.nunique() == 1
    assert tabla.brier_grupo.iloc[0] == pytest.approx((0 + .45**2 + 0) / 3)


def test_validacion_rechaza_holdout_y_modelos_con_registros_distintos():
    df = probabilidades()
    reservado = df.copy()
    reservado.loc[reservado.index[0], "anio_prueba"] = 2025
    with pytest.raises(ValueError, match="reservados"):
        act.validar_probabilidades(reservado)
    distinto = df.copy()
    indice = distinto.index[distinto.modelo.eq("XGBoost")][0]
    distinto.loc[indice, "y_real"] = 1 - distinto.loc[indice, "y_real"]
    with pytest.raises(ValueError, match="mismos registros"):
        act.validar_probabilidades(distinto)


def test_grilla_y_escenarios_fijos():
    df = act.validar_probabilidades(probabilidades())
    tabla = act.tabla_thresholds(df, ("modelo",))
    assert len(act.GRID_THRESHOLDS) == 76
    resumen = tabla.groupby("modelo").threshold.agg(["min", "max", "count"])
    assert resumen["min"].eq(.05).all() and resumen["max"].eq(.80).all()
    assert resumen["count"].eq(76).all()
    escenarios = act.seleccionar_escenarios(tabla)
    assert set(escenarios.escenario) == set(act.ESCENARIOS)
    assert escenarios.groupby("modelo").size().eq(6).all()
    assert escenarios.query("escenario == 'referencia_050'").threshold.eq(.5).all()
    for objetivo, minimo in [("recall_080", .8), ("recall_085", .85), ("recall_090", .9)]:
        elegidos = escenarios.query("escenario == @objetivo and viable")
        assert elegidos.recall.ge(minimo).all()


def test_walk_forward_solo_usa_anios_anteriores():
    df = act.validar_probabilidades(probabilidades())
    resultado = act.validacion_temporal_escenarios(df)
    assert set(resultado.anio_evaluacion) == set(range(2019, 2025))
    assert (resultado.anio_maximo_seleccion < resultado.anio_evaluacion).all()
    assert not resultado.anio_evaluacion.ge(2025).any()
    fila = resultado.query("anio_evaluacion == 2022").iloc[0]
    assert fila.anios_seleccion == "2018,2019,2020,2021"


def test_ejecucion_genera_salidas_sin_recalibrar(tmp_path):
    entrada = tmp_path / "probas.csv"
    salida = tmp_path / "salida"
    original = probabilidades()
    baseline = original.loc[original.modelo.eq(act.MODELOS[0])].copy()
    baseline["modelo"] = "Persistencia_Presion"
    baseline["tipo_score"] = "score_determinista_0_1"
    original = pd.concat([original, baseline], ignore_index=True)
    original.to_csv(entrada, index=False)
    antes = entrada.read_bytes()
    tablas, metodologia = act.ejecutar(entrada, salida)
    assert all((salida / nombre).is_file() for nombre in act.ARCHIVOS)
    assert entrada.read_bytes() == antes
    assert metodologia["2025_utilizado"] is False
    assert metodologia["2026_utilizado"] is False
    assert metodologia["calibrador_ajustado"] is False
    assert metodologia["umbral_operacional_definitivo"] is None
    assert set(tablas["calibracion_por_familia.csv"].familia_servicio) == {
        "hospitalizacion", "cuidados_criticos"}
    cargada = json.loads((salida / "metodologia_calibracion_threshold.json").read_text())
    assert cargada["grid_thresholds"] == {"desde": .05, "hasta": .8, "paso": .01}
    por_anio = tablas["escenarios_operacionales_por_anio.csv"]
    por_familia = tablas["escenarios_operacionales_por_familia.csv"]
    assert set(por_anio.anio_prueba) == set(act.ANIOS_DESARROLLO)
    assert set(por_familia.familia_servicio) == {"hospitalizacion", "cuidados_criticos"}
    assert "brier_score" in por_anio and "brier_score" in por_familia


def test_ejecucion_rechaza_holdout_incluso_si_es_baseline(tmp_path):
    entrada = probabilidades()
    baseline = entrada.iloc[[0]].copy()
    baseline["modelo"] = "Persistencia_Presion"
    baseline["tipo_score"] = "score_determinista_0_1"
    baseline["anio_prueba"] = 2025
    ruta = tmp_path / "probas.csv"
    pd.concat([entrada, baseline]).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="reservados"):
        act.ejecutar(ruta, tmp_path / "salida")
