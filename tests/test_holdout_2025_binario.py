import json

import numpy as np
import pandas as pd
import pytest

from src import evaluar_holdout_2025_binario as h


def _datos():
    filas = []
    periodos = [*pd.period_range("2024-01", "2025-12", freq="M"), pd.Period("2026-01")]
    for predicho in periodos:
        for clase, familia in [(0, 24), (1, 25)]:
            fila = {c: 0.0 for c in h.COLUMNAS_PREDICTORAS_BINARIAS}
            fila.update({"codigo_ipress": f"{familia:08d}",
                         "servicio_hospitalizacion": f"S{familia}",
                         "id_hospitalizacion": f"{familia}0100",
                         "periodo_actual": str(predicho-1), "periodo_predicho": str(predicho),
                         "familia_servicio": "hospitalizacion" if familia == 24 else "cuidados_criticos",
                         "ocupacion_estimada": [.7, .95][clase],
                         "ocupacion_siguiente_mes": [.8, 1.0][clase], h.OBJETIVO: clase})
            filas.append(fila)
    return pd.DataFrame(filas)


def _fuentes(tmp_path):
    dataset, metadata = tmp_path/"dataset.csv", tmp_path/"metadata.json"
    escenarios, probas = tmp_path/"escenarios.csv", tmp_path/"probas.csv"
    _datos().to_csv(dataset, index=False)
    metadata.write_text(json.dumps({
        "dataset_sha256": h._sha(dataset),
        "definicion_target": {"version": h.VERSION_TARGET, "variable_objetivo": h.OBJETIVO,
                              "comparador": ">", "umbral": .90},
        "tratamiento_capacidad": {"version": h.VERSION_POLITICA,
                                  "reglas_aplicadas": ["Q05", "Q06", "Q07", "Q08"]}}))
    pd.DataFrame([{"modelo": m, "escenario": "max_f1", "threshold": t}
                  for m, t in h.THRESHOLDS.items()]).to_csv(escenarios, index=False)
    filas = []
    for anio in range(2018, 2025):
        for indice, y in enumerate([0, 1]):
            for modelo in h.CONFIGURACION:
                filas.append({"indice_dataset": anio*10+indice, "modelo": modelo,
                    "tipo_score": "probabilidad_sin_calibracion", "anio_prueba": anio,
                    "familia_servicio": ["hospitalizacion", "cuidados_criticos"][y],
                    "y_real": y, "score_positivo": [.2, .8][y], "codigo_ipress": "00000001",
                    "servicio_hospitalizacion": "S", "periodo_actual": f"{anio-1}-12",
                    "periodo_predicho": f"{anio}-01"})
    pd.DataFrame(filas).to_csv(probas, index=False)
    return dataset, metadata, escenarios, probas


def _motor():
    class Modelo:
        classes_ = np.array([0, 1])
        def fit(self, X, y):
            assert len(X) == len(y) and set(y) == {0, 1}; return self
        def predict_proba(self, X):
            p = np.where(pd.to_numeric(X.ocupacion_estimada) > .9, .8, .2)
            return np.column_stack([1-p, p])
    return {m: m for m in h.CONFIGURACION}, lambda X, algoritmo: Modelo()


def test_auditoria_separa_por_periodo_objetivo_y_cierra_2026():
    df = _datos().drop(columns=[h.OBJETIVO, "ocupacion_siguiente_mes"])
    train, test, reservado, audit = h.particionar_y_auditar(df)
    assert audit["max_periodo_features_train"] == "2024-11"
    assert audit["max_periodo_predicho_train"] == "2024-12"
    assert audit["min_periodo_predicho_test"] == "2025-01"
    assert audit["max_periodo_predicho_test"] == "2025-12"
    assert audit["interseccion_observaciones_train_test"] == 0
    assert audit["saltos_temporales"] == 0
    assert train.sum() == 24 and test.sum() == 24 and reservado.sum() == 2


def test_preapertura_congela_decisiones_sin_etiquetas(tmp_path):
    fuentes, salida = _fuentes(tmp_path), tmp_path/"holdout"
    audit = h.preauditar(*fuentes, salida)
    assert audit["etiquetas_2025_leidas"] is False
    frozen = json.loads((salida/"congelacion_preholdout_2025.json").read_text())
    assert frozen["modelo_principal"] == "XGBoost" and frozen["threshold_principal"] == .36
    assert frozen["thresholds_sensibilidad"] == [.32, .28]
    with pytest.raises(FileExistsError): h.preauditar(*fuentes, salida)


def test_evaluacion_unica_genera_outputs_y_no_predice_2026(tmp_path):
    fuentes, salida = _fuentes(tmp_path), tmp_path/"holdout"
    h.preauditar(*fuentes, salida); modelos, constructor = _motor()
    h.evaluar(*fuentes, salida, modelos=modelos, constructor_pipeline=constructor,
              repeticiones_bootstrap=20)
    assert all((salida/n).is_file() for n in h.ARCHIVOS_FINALES)
    audit = json.loads((salida/"auditoria_temporal_holdout_2025.json").read_text())
    assert audit["ajuste_terminado_antes_de_abrir_etiquetas"] is True
    assert audit["aperturas_etiqueta_2025"] == 1
    assert audit["metricas_2025_calculadas"] is True
    pred = pd.read_csv(salida/"predicciones_holdout_2025.csv")
    assert set(pred.periodo_predicho.str[:4]) == {"2025"}
    assert set(pd.read_csv(salida/"sensibilidad_thresholds_2025.csv").threshold) == {.36, .32, .28}
    ver = json.loads((salida/"verificacion_holdout_2025.json").read_text())
    assert ver["2026_predicho"] is False and ver["2026_evaluado"] is False
    assert all(h._sha(salida/nombre) == esperado
               for nombre, esperado in ver["archivos"].items())
    with pytest.raises(FileExistsError):
        h.evaluar(*fuentes, salida, modelos=modelos, constructor_pipeline=constructor,
                  repeticiones_bootstrap=2)


def test_rechaza_target_2025_en_desarrollo(tmp_path):
    dataset, metadata, escenarios, probas = _fuentes(tmp_path)
    df = pd.read_csv(probas); df.loc[0, "anio_prueba"] = 2025; df.to_csv(probas, index=False)
    with pytest.raises(ValueError, match="reservado"):
        h.preauditar(dataset, metadata, escenarios, probas, tmp_path/"holdout")


def test_bootstrap_cluster_reproducible():
    y, p, grupos = [0,1,0,1,0,1], [.1,.8,.4,.7,.2,.9], ["a","a","b","b","c","c"]
    a = h.bootstrap_cluster(y, p, grupos, repeticiones=30, semilla=7)
    b = h.bootstrap_cluster(y, p, grupos, repeticiones=30, semilla=7)
    pd.testing.assert_frame_equal(a, b)
    recall = a.set_index("metrica").loc["recall", "estimacion"]
    assert recall == h.metricas_threshold(y, p, .36)["recall"]
