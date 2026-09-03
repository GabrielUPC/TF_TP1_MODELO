import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import optimizar_xgboost as opt


def datos():
    filas = []
    for periodo in pd.period_range('2015-01', '2026-12', freq='M'):
        for clase in range(3):
            filas.append({'codigo_ipress': str(clase), 'servicio_hospitalizacion': 'S',
                          'periodo_actual': str(periodo-1), 'periodo_predicho': str(periodo),
                          'ocupacion_estimada': [.5, .75, .9][clase],
                          'nivel_riesgo_actual_codificado': clase, opt.bt.OBJETIVO: clase})
    return pd.DataFrame(filas)


def motor():
    llamadas = []
    class Algoritmo:
        def __init__(self, **params):
            self.params = params
        def get_params(self):
            return self.params.copy()
        def set_params(self, **params):
            self.params.update(params)
            return self
    class Pipeline:
        classes_ = [0, 1, 2]
        def __init__(self, alg):
            self.alg = alg
        def fit(self, X, y, **kwargs):
            assert X.index.equals(y.index)
            llamadas.append(('fit', X.index.tolist(), kwargs))
            return self
        def predict(self, X):
            llamadas.append(('predict', X.index.tolist(), {}))
            return np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
        def predict_proba(self, X):
            pred = np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
            return np.eye(3)[pred]
    return SimpleNamespace(COLUMNAS_PREDICTORAS=['ocupacion_estimada'],
        COLUMNAS_EXCLUIDAS=list(opt.bt.ETIQUETAS),
        obtener_modelos=lambda: {'XGBoost': Algoritmo(), 'Random_Forest': Algoritmo(tipo='rf')},
        crear_pipeline=lambda X, a: Pipeline(a), llamadas=llamadas)


def test_espacio_limitado_incluye_base_y_nueve_parametros():
    configs = opt.configuraciones()
    assert len(configs) == 15
    assert configs[0]['configuracion'] == opt.BASE
    assert configs[0]['parametros']['n_estimators'] == 300
    assert configs[0]['parametros']['max_depth'] == 5
    assert {c['esquema_pesos'] for c in configs} == set(opt.PESOS)
    for c in configs:
        assert set(opt.PERFILES['base']).issubset(c['parametros'])
        assert c['parametros']['random_state'] == 42


@pytest.mark.parametrize('factor', [1.1, 1.2, 1.3, 1.5])
def test_pesos_absolutos_favorecen_alto(factor):
    y = [0, 0, 0, 1, 1, 2]
    pesos = opt.sample_weights(y, f'alto_{factor}')
    np.testing.assert_allclose(pesos, [1, 1, 1, 1, 1, factor])


def test_balanceado_solo_cuenta_train():
    np.testing.assert_allclose(opt.sample_weights([0, 0, 0, 1, 1, 2], 'balanceado'),
                               [2/3, 2/3, 2/3, 1, 1, 2])


def test_holdout_y_futuro_no_participan_en_tuning():
    df = datos()
    historico, folds, plan = opt.historial_tuning(df)
    assert historico.periodo_predicho.lt('2025-01').all()
    assert plan.anio_prueba.lt(2025).all()
    for anio, train, test in folds:
        assert anio < 2025
        assert df.loc[train, 'periodo_predicho'].lt(f'{anio}-01').all()
        assert df.loc[test, 'periodo_predicho'].str.startswith(str(anio)).all()
    cambiado = df.copy()
    cambiado.loc[cambiado.periodo_predicho.ge('2025-01'), opt.bt.OBJETIVO] = 99
    historico2, folds2, _ = opt.historial_tuning(cambiado)
    pd.testing.assert_frame_equal(historico, historico2)
    assert [f[0] for f in folds] == [f[0] for f in folds2]


def resultados_ranking():
    filas = []
    # base, mal recall, seguridad con F1 permitido, F1 por debajo del límite.
    for nombre, f1, bal, recall in [(opt.BASE, .70, .71, .75), ('malo', .74, .74, .70),
                                   ('seguro', .69, .70, .85), ('degradado', .67, .68, .90),
                                   (opt.PERSISTENCIA, .65, .67, .81)]:
        for anio in [2023, 2024]:
            m = dict.fromkeys(opt.bt.METRICAS, 0.)
            m.update(f1_macro=f1, balanced_accuracy=bal, recall_alto=recall,
                     tasa_falsos_negativos_alto=1-recall)
            filas.append(dict(configuracion=nombre, anio_prueba=anio, test_sha256=str(anio), n_test=30, **m))
    return pd.DataFrame(filas)


def test_ranking_principal_y_seguridad_respetan_restricciones():
    resumen, principal, seguridad = opt.rankings(resultados_ranking())
    assert principal == opt.BASE
    assert seguridad == 'seguro'
    tabla = resumen.set_index('configuracion')
    assert not tabla.loc['malo', 'admisible_principal']
    assert not tabla.loc['degradado', 'admisible_seguridad']


def test_ranking_prioriza_f1_luego_balanced_y_recall():
    r = resultados_ranking()
    r.loc[r.configuracion == 'seguro', ['f1_macro', 'balanced_accuracy']] = [.71, .70]
    assert opt.rankings(r)[1] == 'seguro'
    r.loc[r.configuracion == 'seguro', ['f1_macro', 'balanced_accuracy']] = [.70, .70]
    assert opt.rankings(r)[1] == opt.BASE


@pytest.mark.parametrize('caso', ['holdout', 'sin_base', 'test_distinto'])
def test_ranking_rechaza_comparaciones_invalidas(caso):
    r = resultados_ranking()
    if caso == 'holdout':
        r['anio_prueba'] += 2
    elif caso == 'sin_base':
        r = r[r.configuracion != opt.BASE]
    else:
        r.loc[0, 'test_sha256'] = 'otro'
    with pytest.raises(ValueError):
        opt.rankings(r)


def test_exportacion_seleccion_congelada_holdout_una_vez_y_produccion_intacta(tmp_path, monkeypatch):
    df = datos()
    original = df.copy(deep=True)
    # Reducir solo el espacio del test; la prueba anterior verifica las 15 reales.
    configuraciones = opt.configuraciones()
    monkeypatch.setattr(opt, 'configuraciones', lambda: configuraciones[:2])
    m = motor()
    sentinel = tmp_path/'modelo_ipress.joblib'
    sentinel.write_bytes(b'produccion')
    resultados, resumen, reporte = opt.optimizar(df, tmp_path, motor=m)
    assert resultados.anio_prueba.max() == 2024
    assert reporte['anio_holdout'] == 2025
    assert reporte['es_modelo_final_produccion'] is False
    assert reporte['configuracion_elegida']['configuracion'] == opt.BASE
    assert len(reporte['resultados_holdout']) == 4
    assert len({r['test_sha256'] for r in reporte['resultados_holdout']}) == 1
    test_2025 = df.index[df.periodo_predicho.str.startswith('2025')].tolist()
    # Base elegida: una sola evaluación XGB reutilizada y una de RF.
    assert len([e for e in m.llamadas if e[0] == 'predict' and e[1] == test_2025]) == 2
    for evento, indices, _ in m.llamadas:
        if evento == 'fit':
            assert df.loc[indices, 'periodo_predicho'].lt('2025-01').all()
    with pytest.raises(FileExistsError):
        opt.optimizar(df, tmp_path, motor=m)
    assert sentinel.read_bytes() == b'produccion'
    assert {'configuracion', 'anio_prueba', 'recall_alto'}.issubset(pd.read_csv(tmp_path/'resultados_tuning_xgboost.csv').columns)
    assert {'ranking_principal', 'ranking_seguridad', 'admisible_principal'}.issubset(pd.read_csv(tmp_path/'resumen_tuning_xgboost.csv').columns)
    j = json.loads((tmp_path/'mejor_configuracion_xgboost.json').read_text())
    assert {'configuracion_elegida', 'comparacion_base_tuning', 'comparacion_persistencia_tuning',
            'anios_tuning', 'criterio_seleccion', 'resultados_holdout'}.issubset(j)
    pd.testing.assert_frame_equal(df, original)
