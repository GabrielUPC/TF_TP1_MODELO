import json
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src import modelo_final as mf, entrenar_modelo_final as ef, predecir
from src.indicadores import agregar_indicadores_registro
from src.variables_temporales_experimentales import CONJUNTOS, agregar_features_candidatas
from src.variables_temporales import agregar_variables_temporales
from src.preparar_dataset import crear_riesgo_actual
from test_api import datos_base
from test_features_temporales_experimentales import datos


class PipelineFalso:
    def __init__(self, columnas=("ocupacion_estimada",), p=(.55, .25, .20), clases=(0, 1, 2)):
        self.feature_names_in_ = np.array(columnas)
        self.classes_ = np.array(clases)
        self.p = np.array(p)
        self.named_steps = {"modelo": SimpleNamespace(feature_importances_=np.ones(len(columnas)))}

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))

    def predict(self, X):
        raise AssertionError("No se permite decidir con predict del pipeline.")


@pytest.mark.parametrize("p,esperada", [
    ([.4, .25, .35], 2), ([.05, .05, .9], 2), ([.55, .25, .2], 1),
    ([.50, .151, .349], 1), ([.6, .21, .19], 0), ([.1, .7, .2], 1),
])
def test_regla_limites_probabilidad_final_y_sin_predict(monkeypatch, p, esperada):
    modelo = PipelineFalso(p=p)
    monkeypatch.setattr(predecir, "cargar_artefactos", lambda: (modelo, predecir.CLASES_ESPERADAS))
    monkeypatch.setattr(predecir, "obtener_variables_principales", lambda datos: [])
    copia = modelo.p.copy()
    salida = predecir.predecir_riesgo({"ocupacion_estimada": .8})
    assert salida["nivel_riesgo_codificado"] == esperada
    assert salida["probabilidad"] == p[esperada]
    for c, nombre in enumerate(("bajo", "medio", "alto")):
        assert salida["probabilidades_por_clase"][nombre] == p[c]
        assert salida["probabilidad_riesgo_"+nombre] == p[c]
    np.testing.assert_array_equal(modelo.p, copia)


def test_reordena_clases_sin_modificar_probabilidades():
    m = PipelineFalso(p=(.35, .4, .25), clases=(2, 0, 1))
    p = mf.probabilidades_ordenadas(m, pd.DataFrame({"ocupacion_estimada": [.8]}))
    np.testing.assert_array_equal(p, [[.4, .25, .35]])
    assert mf.decidir_clases(p).tolist() == [2]


@pytest.mark.parametrize("clases", [(0, 1), (0, 1, 1), (1, 2, 3), ("0", "1", "2")])
def test_rechaza_mapeo_invalido(clases):
    with pytest.raises(ValueError, match="classes_"):
        mf.probabilidades_ordenadas(PipelineFalso(clases=clases), pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize("p", [(np.nan, .2, .8), (-.1, .3, .8), (.4, .5, .7), (.5, .5)])
def test_rechaza_probabilidades_invalidas(p):
    with pytest.raises(ValueError):
        mf.probabilidades_ordenadas(PipelineFalso(p=p), pd.DataFrame({"x": [1]}))


def test_no_inventa_probabilidades_si_falta_predict_proba():
    with pytest.raises(ValueError, match="predict_proba"):
        mf.probabilidades_ordenadas(SimpleNamespace(classes_=[0, 1, 2]), pd.DataFrame({"x": [1]}))


def test_contrato_d_rechaza_columnas_extra_faltantes_y_orden():
    columnas = ["ocupacion_estimada", *CONJUNTOS['D']]
    modelo = mf.ModeloFinalD(PipelineFalso(columnas))
    X = pd.DataFrame(np.zeros((2, len(columnas))), columns=columnas)
    assert modelo.predict_proba(X).shape == (2, 3)
    assert modelo.predict(X).tolist() == [1, 1]
    for incorrecto in (X.assign(extra=1), X.iloc[:, :-1], X[X.columns[::-1]]):
        with pytest.raises(ValueError, match="exactamente"):
            modelo.predict_proba(incorrecto)


def test_entrada_api_reproduce_d_y_no_modifica_historia():
    historia = [agregar_indicadores_registro(datos_base(2024, m)) for m in range(1, 7)]
    original = json.dumps(historia, sort_keys=True)
    esperado, _ = crear_riesgo_actual(agregar_variables_temporales(pd.DataFrame(historia)))
    esperado = agregar_features_candidatas(esperado)
    fila, completo = mf.preparar_entrada_d(historia[-1], historia[:-1])
    assert completo
    np.testing.assert_allclose(fila[CONJUNTOS['D']], esperado.iloc[[-1]][CONJUNTOS['D']], equal_nan=True)
    assert json.dumps(historia, sort_keys=True) == original


def test_entrada_d_hueco_no_imputa_ni_puentea():
    actual = agregar_indicadores_registro(datos_base(2024, 6))
    previos = [agregar_indicadores_registro(datos_base(2024, m)) for m in (1, 2, 3, 4)]
    fila, completo = mf.preparar_entrada_d(actual, previos)
    assert not completo
    assert pd.isna(fila.iloc[0].ocupacion_lag_1m)
    assert pd.isna(fila.iloc[0].promedio_ocupacion_6m)


@pytest.fixture
def repositorio(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    df = datos("2024-01", "2024-06")
    csv = tmp_path / "data/processed/dataset_modelo_ipress.csv"
    df.to_csv(csv, index=False)
    sha = ef.sha256_archivo(csv)
    (csv.parent / "dataset_metadata.json").write_text(json.dumps({"dataset_sha256": sha,
        "tratamiento_capacidad": {"version": ef.VERSION_POLITICA}}))
    columnas = ["ocupacion_estimada", *CONJUNTOS['D']]
    seleccion = {"estado": "completado_sin_produccion", "regla_seleccionada": mf.REGLA_FINAL['nombre'],
        "anios_desarrollo": mf.ANIOS_DESARROLLO, "procedencia": {"dataset_sha256": sha},
        "columnas_predictoras": columnas, "hiperparametros": {"random_state": 42}}
    (tmp_path / "models/seleccion_regla_extension_020.json").write_text(json.dumps(seleccion))
    filas = [{"regla": mf.REGLA_FINAL['nombre'], "anio_prueba": a,
              **{m: .75 for m in ef.bt.METRICAS}} for a in [*mf.ANIOS_DESARROLLO, 2025]]
    pd.DataFrame(filas).to_csv(tmp_path / "models/resultados_reglas_extension_020.csv", index=False)
    pd.DataFrame([{"regla": mf.REGLA_FINAL['nombre'], **{m+"_promedio": .75 for m in ef.bt.METRICAS}}]).to_csv(
        tmp_path / "models/resumen_reglas_extension_020.csv", index=False)
    (tmp_path / "models/clases_riesgo.json").write_text(json.dumps(predecir.CLASES_ESPERADAS))
    (tmp_path / "models/modelo_ipress.joblib").write_bytes(b"anterior")
    (tmp_path / "models/model_metadata.json").write_text(json.dumps({"metricas_temporales": {"f1_macro": .1}}))
    for nombre in ("entrenar_modelo_final.py", "entrenar_modelo.py", "modelo_final.py",
                   "variables_temporales_experimentales.py", "variables_temporales.py"):
        (tmp_path / "src" / nombre).write_text("codigo sintético")
    return tmp_path


def motor_falso(monkeypatch, eventos):
    def ajustar(nombre, pipeline, X, y):
        eventos.append((nombre, len(X), list(X.columns)))
        return pipeline
    motor = SimpleNamespace(COLUMNAS_PREDICTORAS=["ocupacion_estimada"],
        obtener_modelos=lambda: {"XGBoost": SimpleNamespace(get_params=lambda: {"random_state": 42})},
        crear_pipeline=lambda X, algoritmo: PipelineFalso(X.columns), ajustar_pipeline=ajustar,
        _mapear_importancias_transformadas=lambda p, i: pd.DataFrame({"variable": p.feature_names_in_, "importancia": i}))
    monkeypatch.setitem(sys.modules, "src.entrenar_modelo", motor)
    import src
    monkeypatch.setattr(src, "entrenar_modelo", motor, raising=False)
    monkeypatch.setattr(ef.importlib.metadata, "version", lambda _: "prueba")
    return motor


def test_entrenamiento_unico_metadata_y_evidencia_intacta(repositorio, monkeypatch):
    eventos = []
    motor_falso(monkeypatch, eventos)
    antes = ef.huellas_protegidas(repositorio)
    metadata = ef.entrenar_final(repositorio)
    assert len(eventos) == 1 and eventos[0][0] == "XGBoost" and eventos[0][1] == 18
    assert ef.huellas_protegidas(repositorio) == antes
    assert metadata['conjunto_features'] == 'D' and metadata['regla_decision'] == mf.REGLA_FINAL
    assert metadata['es_modelo_final_produccion'] is True
    assert metadata['anios_desarrollo'] == mf.ANIOS_DESARROLLO
    assert "no holdout virgen" in metadata['advertencia_2025']
    assert 'metricas_temporales' not in metadata
    assert metadata['metricas_historicas_no_vigentes']['metricas_temporales']['f1_macro'] == .1
    assert set(metadata['pesos_por_clase'].values()) == {1.0}
    modelo = joblib.load(repositorio / "models/modelo_ipress.joblib")
    assert list(modelo.feature_names_in_) == metadata['lista_features']
    assert metadata['modelo_sha256'] == ef.sha256_archivo(repositorio / "models/modelo_ipress.joblib")
    assert not list((repositorio / "models").glob('.final_*'))


def test_error_entrenando_preserva_artefactos_anteriores(repositorio, monkeypatch):
    motor = motor_falso(monkeypatch, [])
    def fallar(*args):
        raise RuntimeError("fallo de ajuste")
    motor.ajustar_pipeline = fallar
    antes = {p: p.read_bytes() for p in (repositorio / 'models').iterdir()}
    with pytest.raises(RuntimeError, match="fallo de ajuste"):
        ef.entrenar_final(repositorio)
    assert all(p.read_bytes() == b for p, b in antes.items())


def test_error_publicando_revierte_par(repositorio, monkeypatch):
    motor_falso(monkeypatch, [])
    antes = {p: p.read_bytes() for p in (repositorio / 'models').iterdir()}
    replace = ef.os.replace
    def fallar(src, dst):
        if Path(dst).name == 'model_metadata.json':
            raise OSError('fallo de reemplazo')
        return replace(src, dst)
    from pathlib import Path
    monkeypatch.setattr(ef.os, 'replace', fallar)
    with pytest.raises(OSError, match='fallo de reemplazo'):
        ef.entrenar_final(repositorio)
    assert all(p.read_bytes() == b for p, b in antes.items())


def test_plan_no_es_produccion_ni_escribe(repositorio):
    antes = {p: p.read_bytes() for p in repositorio.rglob('*') if p.is_file()}
    _, _, plan = ef.leer_plan(repositorio)
    assert plan['es_modelo_final_produccion'] is False
    assert all(p.read_bytes() == b for p, b in antes.items())


def test_rechaza_dataset_distinto(repositorio):
    with (repositorio/'data/processed/dataset_modelo_ipress.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='huellas'):
        ef.leer_plan(repositorio)


def test_carga_productiva_verifica_metadata_y_hash(repositorio, monkeypatch):
    motor_falso(monkeypatch, [])
    ef.entrenar_final(repositorio)
    monkeypatch.setattr(predecir, 'MODEL_PATH', repositorio/'models/modelo_ipress.joblib')
    monkeypatch.setattr(predecir, 'METADATA_PATH', repositorio/'models/model_metadata.json')
    monkeypatch.setattr(predecir, 'CLASES_PATH', repositorio/'models/clases_riesgo.json')
    predecir.limpiar_caches()
    try:
        modelo, _ = predecir.cargar_artefactos()
        assert modelo.regla_decision_ == mf.REGLA_FINAL
        meta = predecir.cargar_metadata()
        meta['modelo_sha256'] = 'incorrecto'
        predecir.METADATA_PATH.write_text(json.dumps(meta))
        predecir.limpiar_caches()
        with pytest.raises(RuntimeError, match='contrato final'):
            predecir.cargar_artefactos()
    finally:
        predecir.limpiar_caches()


def evidencia_baselines(repositorio, f1_baseline):
    _, _, metadata = ef.leer_plan(repositorio)
    ruta = repositorio/'models/resultados_reglas_extension_020.csv'
    resultados = pd.read_csv(ruta)
    resultados['n_train'] = 100
    resultados['n_test'] = 30
    resultados['test_sha256'] = 'test_' + resultados.anio_prueba.astype(str)
    resultados.to_csv(ruta, index=False)
    filas = []
    for nombre in ('Clase_Mayoritaria', 'Persistencia_Riesgo_Actual', 'Regla_Ocupacion_Actual'):
        base = resultados.copy()
        base['modelo'], base['tipo'] = nombre, 'baseline'
        base['f1_macro'] = f1_baseline
        # Un baseline perfecto en 2025 no debe influir en esta comparación.
        base.loc[base.anio_prueba.eq(2025), 'f1_macro'] = 1.0
        filas.append(base)
    pd.concat(filas).to_csv(repositorio/'models/metricas_backtesting_temporal.csv', index=False)
    (repositorio/'models/comparacion_backtesting_temporal.json').write_text(json.dumps({
        'dataset_sha256': metadata['dataset_sha256']}))
    return metadata


@pytest.mark.parametrize('baseline,esperado', [(.74, False), (.73, False), (.70, True)])
def test_supera_baseline_respeta_margen_y_excluye_2025(repositorio, baseline, esperado):
    metadata = evidencia_baselines(repositorio, baseline)
    metadata['metricas_historicas_no_vigentes'] = {'supera_baseline': not esperado}
    valor, evidencia = ef.evaluar_supera_baseline(metadata, repositorio)
    assert valor is esperado
    assert evidencia['evidencia_verificada'] is True
    assert evidencia['f1_macro_regla_final_promedio'] == .75
    assert evidencia['mejor_f1_macro_baseline_promedio'] == pytest.approx(baseline)
    assert evidencia['margen_f1_requerido'] == .02


@pytest.mark.parametrize('problema', ['hash', 'dataset', 'faltante', 'regla'])
def test_sin_evidencia_comparable_no_afirma_superioridad(repositorio, problema):
    metadata = evidencia_baselines(repositorio, .1)
    ruta = repositorio/'models/metricas_backtesting_temporal.csv'
    if problema == 'hash':
        df = pd.read_csv(ruta)
        df.loc[0, 'test_sha256'] = 'otro_test'
        df.to_csv(ruta, index=False)
    elif problema == 'dataset':
        metadata['dataset_sha256'] = 'otro_dataset'
    elif problema == 'faltante':
        ruta.unlink()
    else:
        metadata['regla_decision'] = {'nombre': 'argmax'}
    valor, evidencia = ef.evaluar_supera_baseline(metadata, repositorio)
    assert valor is False and evidencia['evidencia_verificada'] is False
    assert 'No se puede demostrar' in evidencia['motivo']


def test_cli_actualiza_solo_metadata_sin_entrenar(repositorio, monkeypatch):
    metadata = evidencia_baselines(repositorio, .74)
    metadata.update(es_modelo_final_produccion=True,
                    modelo_sha256=ef.sha256_archivo(repositorio/'models/modelo_ipress.joblib'))
    ruta = repositorio/'models/model_metadata.json'
    ruta.write_text(json.dumps(metadata))
    antes = {p: p.read_bytes() for p in repositorio.rglob('*') if p.is_file() and p != ruta}
    monkeypatch.setattr(sys, 'argv', ['entrenar_modelo_final', '--actualizar-solo-metadata'])
    actualizar = ef.actualizar_solo_metadata
    monkeypatch.setattr(ef, 'actualizar_solo_metadata', lambda: actualizar(repositorio))
    def prohibido(*args, **kwargs):
        raise AssertionError('No se permite entrenar, publicar ni cargar joblib.')
    monkeypatch.setattr(ef, 'entrenar_final', prohibido)
    monkeypatch.setattr(ef, 'publicar', prohibido)
    monkeypatch.setattr(ef.joblib, 'load', prohibido)
    ef.main()
    nuevo = json.loads(ruta.read_text(encoding='utf-8'))
    assert nuevo.pop('supera_baseline') is False
    assert nuevo.pop('comparacion_baselines_vigente')['evidencia_verificada'] is True
    assert nuevo == metadata
    assert all(p.read_bytes() == contenido for p, contenido in antes.items())
