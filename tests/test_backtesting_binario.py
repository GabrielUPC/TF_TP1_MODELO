import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src import backtesting_binario as bt


def datos(desde="2018-01", hasta="2026-03"):
    filas = []
    for periodo in pd.period_range(desde, hasta, freq="M"):
        for familia in (24, 25):
            for clase in (0, 1):
                fila = {c: 0.0 for c in bt.COLUMNAS_PREDICTORAS_BINARIAS}
                fila.update({"codigo_ipress": f"{familia*10+clase:08d}",
                             "servicio_hospitalizacion": "SERVICIO",
                             "id_hospitalizacion": f"{familia}0100",
                             "periodo_actual": str(periodo-1), "periodo_predicho": str(periodo),
                             "ocupacion_estimada": [.90, .91][clase],
                             "ocupacion_siguiente_mes": [.90, .91][clase],
                             bt.OBJETIVO: clase})
                filas.append(fila)
    df = pd.DataFrame(filas)
    df.index = df.index*7+13
    return df


def motor_prueba():
    eventos = []

    class Modelo:
        classes_ = np.array([1, 0])

        def __init__(self, nombre):
            self.nombre = nombre

        def fit(self, X, y):
            assert len(X) == len(y)
            eventos.append(("fit", self.nombre, X.index.tolist(), list(X.columns)))
            return self

        def predict_proba(self, X):
            eventos.append(("predict", self.nombre, X.index.tolist()))
            p = np.where(X.ocupacion_estimada.gt(.90), .8, .2)
            return np.column_stack([p, 1-p])

    def constructor(X, algoritmo):
        eventos.append(("crear", algoritmo, X.index.tolist()))
        return Modelo(algoritmo)

    return {n: n for n in bt.CONFIGURACION}, constructor, eventos


def test_folds_solo_desarrollo_sin_futuro_expansivos():
    df = datos()
    folds, plan = bt.crear_folds_binarios(df)
    assert [a for a, _, _ in folds] == [2020, 2021, 2022, 2023, 2024]
    anterior = set()
    for anio, train, test in folds:
        anios_train = pd.PeriodIndex(df.loc[train, "periodo_predicho"], freq="M").year
        anios_test = pd.PeriodIndex(df.loc[test, "periodo_predicho"], freq="M").year
        assert (anios_train < anio).all()
        assert (anios_test == anio).all() and anio < 2025
        assert set(train).isdisjoint(test) and anterior < set(train)
        anterior = set(train)
    assert not plan.loc[plan.anio_prueba.ge(2025), "elegible"].any()
    assert plan.loc[plan.anio_prueba.ge(2025), "train_clase_1"].isna().all()


def test_holdout_se_cuenta_sin_inspeccionar_etiquetas():
    df = datos()
    df.loc[df.periodo_predicho.ge("2025-01"), bt.OBJETIVO] = 999
    plan = bt.planificar_folds_binarios(df)
    assert plan.loc[plan.anio_prueba.eq(2025), "n_test"].item() == 48
    assert plan.loc[plan.anio_prueba.eq(2026), "n_test"].item() == 12
    assert not plan.loc[plan.anio_prueba.ge(2025), "elegible"].any()


def test_23_meses_no_bastan_y_24_con_hueco_si():
    df = datos("2018-02", "2020-12")
    assert not bt.planificar_folds_binarios(df).query("anio_prueba == 2020").elegible.item()
    extra = datos("2017-12", "2017-12")
    df = pd.concat([extra, df], ignore_index=True)
    plan = bt.planificar_folds_binarios(df).query("anio_prueba == 2020")
    assert plan.elegible.item() and plan.meses_train.item() == 24
    assert plan.huecos_historial.item() == 1


def test_anio_incompleto_y_train_sin_dos_clases_no_elegibles():
    df = datos().query("periodo_predicho != '2021-05'")
    df.loc[df.periodo_predicho.lt("2020-01"), bt.OBJETIVO] = 0
    plan = bt.planificar_folds_binarios(df)
    assert "cada clase" in plan.query("anio_prueba == 2020").motivo.item()
    assert "incompleto" in plan.query("anio_prueba == 2021").motivo.item()
    assert plan.query("anio_prueba == 2022").elegible.item()


@pytest.mark.parametrize("min_meses,min_casos", [(23, 2), (24, 1)])
def test_minimos_no_se_relajan(min_meses, min_casos):
    with pytest.raises(ValueError, match="Minimos"):
        bt.planificar_folds_binarios(datos(), min_meses, min_casos)


@pytest.mark.parametrize("caso", ["salto", "duplicado", "indice", "clase"])
def test_estructura_invalida_rechazada(caso):
    df = datos()
    if caso == "salto":
        df.loc[df.index[0], "periodo_predicho"] = "2018-02"
    elif caso == "duplicado":
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    elif caso == "indice":
        df.index = [0]*len(df)
    else:
        df.loc[df.index[0], bt.OBJETIVO] = 2
    with pytest.raises(ValueError):
        bt.crear_folds_binarios(df)


def test_metricas_binarias_auc_ap_y_confusion():
    m = bt.calcular_metricas_binarias([0, 0, 1, 1], [.1, .4, .35, .8])
    assert m["roc_auc"] == .75
    assert m["average_precision"] == pytest.approx(5/6)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 2, 0, 1)
    assert m["recall"] == .5 and m["specificity"] == 1
    assert m["precision"] == 1 and m["f1"] == pytest.approx(2/3)
    assert m["balanced_accuracy"] == .75 and m["accuracy"] == .75
    assert m["brier_score"] == pytest.approx(.158125)
    assert m["prevalencia_real"] == .5 and m["prevalencia_predicha"] == .25
    assert m["tasa_falsos_negativos"] == .5


def test_ap_empates_y_metricas_sin_ambas_clases():
    m = bt.calcular_metricas_binarias([1, 0, 1, 0], [.4]*4)
    assert m["roc_auc"] == .5 and m["average_precision"] == .5
    m = bt.calcular_metricas_binarias([0, 0], [.1, .2])
    assert np.isnan(m["roc_auc"]) and np.isnan(m["average_precision"])
    assert np.isnan(m["recall"]) and np.isnan(m["tasa_falsos_negativos"])
    assert m["specificity"] == 1
    m = bt.calcular_metricas_binarias([1, 1], [.1, .2])
    assert np.isnan(m["specificity"]) and m["average_precision"] == 1


@pytest.mark.parametrize("score", [[1.1, .1], [np.nan, .1], [.1], [[.1], [.2]]])
def test_scores_invalidos_rechazados(score):
    with pytest.raises(ValueError):
        bt.calcular_metricas_binarias([0, 1], score)


def test_no_cambia_umbral_tecnico_y_calibracion_diez_bins():
    with pytest.raises(ValueError, match="umbral"):
        bt.calcular_metricas_binarias([0, 1], [.1, .9], umbral=.4)
    c = bt.tabla_calibracion([0, 1, 1], [0, .5, 1])
    assert len(c) == 10 and c.n.sum() == 3
    assert c.loc[c.bin.eq(0), "probabilidad_media"].item() == 0
    assert c.loc[c.bin.eq(9), "probabilidad_media"].item() == 1
    assert c.loc[c.n.eq(0), "frecuencia_observada"].isna().all()


def test_mismos_registros_features_y_outputs_sin_holdout_ni_modelo(tmp_path):
    df, (modelos, constructor, eventos) = datos(), motor_prueba()
    df.loc[df.periodo_predicho.ge("2025-01"), bt.OBJETIVO] = 999
    original = df.copy(deep=True)
    sentinel = tmp_path/"modelo_ipress.joblib"
    sentinel.write_bytes(b"produccion protegida")
    metricas, resumen, estratos, criterio = bt.ejecutar_backtesting_binario(
        df, tmp_path, modelos=modelos, constructor_pipeline=constructor)
    assert len(metricas) == 25 and len(resumen) == 5 and len(estratos) == 50
    for anio, train, test in bt.crear_folds_binarios(df)[0]:
        grupo = metricas.query("anio_prueba == @anio")
        assert grupo.test_sha256.nunique() == 1 and grupo.n_test.nunique() == 1
        for nombre in modelos:
            assert ("crear", nombre, train.tolist()) in eventos
            assert ("fit", nombre, train.tolist(), bt.COLUMNAS_PREDICTORAS_BINARIAS) in eventos
            assert ("predict", nombre, test.tolist()) in eventos
    assert metricas.query("tipo == 'modelo'").roc_auc.eq(1).all()
    assert metricas.query("modelo == 'Persistencia_Presion'").fn.eq(0).all()
    assert set(estratos.familia_servicio) == {"hospitalizacion", "cuidados_criticos"}
    assert estratos.n.eq(24).all() and estratos.positivos.eq(12).all()
    assert criterio["holdout_evaluado"] is False and criterio["2026_utilizado"] is False
    assert criterio["sample_weight"] is None and criterio["class_weight"] is None
    assert resumen.query("tipo == 'baseline'").ranking_candidato.isna().all()
    p = pd.read_csv(tmp_path/"probabilidades_desarrollo_binario.csv")
    assert p.anio_prueba.max() == 2024 and p.y_real.isin([0, 1]).all()
    assert {"score_determinista_0_1", "probabilidad_sin_calibracion"} == set(p.tipo_score)
    assert {"tp", "tn", "fp", "fn"}.issubset(pd.read_csv(tmp_path/"matrices_confusion_binario.csv").columns)
    assert {"average_precision_promedio", "roc_auc_std", "f1_min", "brier_score_max"}.issubset(resumen.columns)
    assert json.loads((tmp_path/"criterio_comparacion_binario.json").read_text())["anios_desarrollo"] == [2020, 2021, 2022, 2023, 2024]
    assert (tmp_path/"resumen_metodologico.md").is_file()
    assert (tmp_path/"auditoria_features.csv").is_file()
    assert sentinel.read_bytes() == b"produccion protegida"
    assert len(list(tmp_path.glob("*.joblib"))) == 1
    pd.testing.assert_frame_equal(df, original)


def test_salida_interna_o_normalizada_hacia_repo_rechazada():
    for path in [bt.ROOT, bt.ROOT/"models", bt.ROOT/"data"/".."/"models"]:
        with pytest.raises(ValueError, match="fuera del repositorio"):
            bt.validar_salida_externa(path)


def test_salida_symlink_hacia_repo_rechazada(tmp_path):
    enlace = tmp_path/"enlace"
    try:
        enlace.symlink_to(bt.ROOT, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("El sistema no permite crear symlinks sin privilegios.")
    with pytest.raises(ValueError, match="fuera del repositorio"):
        bt.validar_salida_externa(enlace/"models")


def test_features_faltantes_impiden_ajuste(tmp_path):
    modelos, constructor, eventos = motor_prueba()
    with pytest.raises(ValueError, match="Faltan features"):
        bt.ejecutar_backtesting_binario(datos().drop(columns=[bt.COLUMNAS_PREDICTORAS_BINARIAS[0]]),
                                      tmp_path, modelos=modelos, constructor_pipeline=constructor)
    assert not eventos


def test_requiere_los_tres_modelos(tmp_path):
    with pytest.raises(ValueError, match="exactamente"):
        bt.ejecutar_backtesting_binario(datos(), tmp_path, modelos={"XGBoost": object()})


@pytest.mark.parametrize("caso", ["etiqueta", "ocupacion_negativa", "ocupacion_ausente"])
def test_api_directa_rechaza_target_incoherente_antes_de_ajustar(tmp_path, caso):
    df = datos()
    indice = df.index[0]
    if caso == "etiqueta":
        df.loc[indice, bt.OBJETIVO] = 1
    elif caso == "ocupacion_negativa":
        df.loc[indice, "ocupacion_siguiente_mes"] = -.01
    else:
        df["ocupacion_siguiente_mes"] = df["ocupacion_siguiente_mes"].astype("Float64")
        df.loc[indice, "ocupacion_siguiente_mes"] = pd.NA
    modelos, constructor, eventos = motor_prueba()
    with pytest.raises(ValueError, match="target de desarrollo"):
        bt.ejecutar_backtesting_binario(df, tmp_path, modelos=modelos,
                                      constructor_pipeline=constructor)
    assert not eventos
    assert not (tmp_path/"criterio_comparacion_binario.json").exists()


def test_preprocesamiento_aprende_solo_train_y_tolera_categoria_nueva():
    from sklearn.dummy import DummyClassifier
    train = pd.DataFrame({"numero": [1.0, 3.0, np.nan], "categoria": ["a", "a", np.nan]})
    test = pd.DataFrame({"numero": [1000.0, np.nan], "categoria": ["futura", np.nan]})
    modelo = bt.crear_pipeline_binario(train, DummyClassifier(strategy="prior"))
    modelo.fit(train, [0, 1, 0])
    modelo.predict_proba(test)
    transformadores = modelo.named_steps["preprocesamiento"].named_transformers_
    assert transformadores["numericas"].named_steps["imputar"].statistics_.tolist() == [2.0]
    assert transformadores["numericas"].named_steps["escalar"].mean_.tolist() == [2.0]
    assert transformadores["categoricas"].named_steps["imputar"].statistics_.tolist() == ["a"]
    assert transformadores["categoricas"].named_steps["codificar"].categories_[0].tolist() == ["a"]


def test_modelos_no_alteran_prevalencia_y_xgb_es_binario():
    modelos = bt.obtener_modelos_binarios()
    assert set(modelos) == set(bt.CONFIGURACION)
    for modelo in modelos.values():
        parametros = modelo.get_params()
        assert parametros.get("class_weight") is None
        assert parametros.get("scale_pos_weight") in (None, 1)
        assert parametros["random_state"] == 42
    assert modelos["XGBoost"].get_params()["objective"] == "binary:logistic"


def test_ranking_prioriza_ap_sobre_accuracy_y_std_desempata():
    filas, estratos = [], []
    for nombre, aps in {"A": [.8, .6], "B": [.71, .71], "C": [.7, .7]}.items():
        for anio, ap in zip([2023, 2024], aps):
            m = bt.calcular_metricas_binarias([0, 1], [.2, .8])
            m.update({"modelo": nombre, "tipo": "modelo", "anio_prueba": anio,
                      "n_test": 2, "test_sha256": str(anio), "average_precision": ap,
                      "accuracy": 1 if nombre == "A" else .5})
            filas.append(m)
            for familia in ["hospitalizacion", "cuidados_criticos"]:
                estratos.append({"modelo": nombre, "familia_servicio": familia,
                                 "average_precision": ap})
    resumen = bt.resumir_modelos_binarios(pd.DataFrame(filas), pd.DataFrame(estratos))
    assert resumen.modelo.tolist() == ["B", "C", "A"]
    assert resumen.ranking_candidato.tolist() == [1, 2, 3]


def metadata_para(df, tmp_path):
    dataset, metadata = tmp_path/"dataset.csv", tmp_path/"metadata.json"
    df.to_csv(dataset, index=False)
    contenido = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                 "tratamiento_capacidad": {"version": bt.VERSION_POLITICA,
                                           "reglas_aplicadas": ["Q05", "Q06", "Q07", "Q08"]},
                 "definicion_target": {"version": bt.VERSION_TARGET, "tipo": "clasificacion_binaria",
                                       "variable_objetivo": bt.OBJETIVO, "indicador": "ocupacion_siguiente_mes",
                                       "formula_ocupacion": "total_pacientes_camas / total_camas_disponibles",
                                       "umbral": .90, "comparador": ">"}}
    metadata.write_text(json.dumps(contenido), encoding="utf-8")
    return dataset, metadata, contenido


def test_carga_valida_hash_politica_target_y_conserva_identidad(tmp_path):
    dataset, metadata, _ = metadata_para(datos(), tmp_path)
    df, procedencia = bt.cargar_dataset_validado(dataset, metadata)
    assert df.codigo_ipress.str.fullmatch(r"[0-9]{8}").all()
    assert len(procedencia["metadata_sha256"]) == 64


@pytest.mark.parametrize("campo", ["hash", "politica", "comparador", "version", "etiqueta"])
def test_carga_rechaza_metadata_o_target_incoherentes(tmp_path, campo):
    df = datos()
    if campo == "etiqueta":
        df.loc[df.index[0], bt.OBJETIVO] = 1
    dataset, metadata, contenido = metadata_para(df, tmp_path)
    if campo == "hash":
        contenido["dataset_sha256"] = "antiguo"
    elif campo == "politica":
        contenido["tratamiento_capacidad"]["version"] = "antigua"
    elif campo == "comparador":
        contenido["definicion_target"]["comparador"] = ">="
    elif campo == "version":
        contenido["definicion_target"]["version"] = "multiclase"
    metadata.write_text(json.dumps(contenido), encoding="utf-8")
    with pytest.raises(ValueError):
        bt.cargar_dataset_validado(dataset, metadata)


def test_cli_solo_plan_no_entrena(tmp_path, monkeypatch):
    dataset, metadata, _ = metadata_para(datos(), tmp_path)
    def prohibido(*args, **kwargs):
        pytest.fail("solo-plan no debe ajustar modelos")
    monkeypatch.setattr(bt, "ejecutar_backtesting_binario", prohibido)
    salida = tmp_path/"salida"
    assert bt.main(["--dataset", str(dataset), "--metadata", str(metadata),
                    "--output-dir", str(salida), "--solo-plan"]) == 0
    assert (salida/"plan_folds_binario.csv").exists()
    assert not (salida/"metricas_backtesting_binario.csv").exists()


def test_cli_exige_destino_externo():
    with pytest.raises(SystemExit) as exc:
        bt.main([])
    assert exc.value.code == 2
