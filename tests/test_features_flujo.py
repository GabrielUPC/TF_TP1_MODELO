import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import evaluar_features_flujo as ev
from src import variables_flujo as vf


def pequeno():
    return pd.DataFrame({'codigo_ipress': ['01']*4, 'servicio_hospitalizacion': ['S']*4,
        'anio': [2020]*4, 'mes': [1, 2, 3, 4], 'total_ingresos': [10., 20., 40., 80.],
        'total_egresos': [5., 25., 30., 50.], 'total_estancias': [10., 75., 120., 250.],
        'total_pacientes_camas': [100., 120., 180., 200.], 'ocupacion_estimada': [.70, .76, .82, .80]})


def test_formulas_exactas_y_unicamente_17_variables():
    d = pequeno()
    r = vf.agregar_features_flujo(d)
    assert len(vf.FEATURES) == len(vf.definiciones()) == 17
    assert set(r)-set(d) == set(vf.FEATURES)
    esperado = {'balance_flujo_mes': 10., 'balance_flujo_acumulado_3m': 10.,
        'promedio_balance_flujo_3m': 10/3, 'meses_ingresos_mayor_egresos_3m': 2,
        'ratio_egresos_ingresos': .75, 'crecimiento_ingresos_1m': 1.,
        'crecimiento_ingresos_2m': 3., 'crecimiento_pacientes_cama_1m': .5,
        'crecimiento_pacientes_cama_2m': .8, 'estancia_promedio_actual': 4.,
        'estancia_promedio_lag_1m': 3., 'cambio_estancia_promedio_1m': 1.,
        'promedio_estancia_3m': 3., 'aceleracion_ingresos': 10.,
        'aceleracion_pacientes_cama': 40., 'racha_ocupacion_creciente_3m': 2,
        'meses_ocupacion_sobre_80_ultimos_3m': 1}
    for c, valor in esperado.items():
        assert r.loc[2, c] == pytest.approx(valor), c
    assert r.loc[3, 'racha_ocupacion_creciente_3m'] == 0
    assert r.loc[3, 'meses_ocupacion_sobre_80_ultimos_3m'] == 1
    pd.testing.assert_frame_equal(r[d.columns], d)


@pytest.mark.parametrize('ocupaciones,racha', [([.7, .76, .82], 2), ([.8, .7, .82], 1),
    ([.7, .82, .82], 0), ([.9, .8, .7], 0), ([.8, .8, .81], 1)])
def test_racha_incrementos_consecutivos_estrictos(ocupaciones, racha):
    d = pequeno().iloc[:3].copy()
    d.ocupacion_estimada = ocupaciones
    assert vf.agregar_features_flujo(d).iloc[-1].racha_ocupacion_creciente_3m == racha


def test_sin_futuro_y_orden_indices_grupos_preservados():
    d = pequeno()
    antes = vf.agregar_features_flujo(d)
    d.loc[3, vf.FUENTES] = 999
    despues = vf.agregar_features_flujo(d)
    pd.testing.assert_frame_equal(antes.iloc[:3], despues.iloc[:3])
    otro = pequeno().assign(servicio_hospitalizacion='T', total_ingresos=999.)
    combinado = pd.concat([pequeno(), otro], ignore_index=True).sample(frac=1, random_state=42)
    r = vf.agregar_features_flujo(combinado)
    assert r.index.equals(combinado.index)
    np.testing.assert_allclose(r.loc[r.servicio_hospitalizacion.eq('S')].sort_index()[vf.FEATURES],
                               antes[vf.FEATURES], equal_nan=True)


def test_huecos_no_se_saltan_y_aceleracion_requiere_tres_meses():
    d = pequeno().drop(index=1)
    r = vf.agregar_features_flujo(d)
    for c in ['balance_flujo_acumulado_3m', 'crecimiento_ingresos_2m',
              'promedio_estancia_3m', 'aceleracion_ingresos', 'aceleracion_pacientes_cama',
              'racha_ocupacion_creciente_3m']:
        assert r[c].isna().all(), c
    assert pd.isna(r.loc[2, 'crecimiento_ingresos_1m'])
    assert r.loc[3, 'crecimiento_ingresos_1m'] == 1.


def test_divisiones_cero_negativos_invalidos_y_ventanas_incompletas_nan():
    d = pequeno()
    d.loc[1, ['total_ingresos', 'total_egresos', 'total_pacientes_camas']] = 0
    r = vf.agregar_features_flujo(d)
    assert pd.isna(r.loc[1, 'ratio_egresos_ingresos'])
    assert pd.isna(r.loc[1, 'estancia_promedio_actual'])
    assert pd.isna(r.loc[2, 'promedio_estancia_3m'])
    assert r.loc[2, ['crecimiento_ingresos_1m', 'crecimiento_pacientes_cama_1m']].isna().all()
    d.loc[2, 'total_estancias'] = np.inf
    d.loc[2, 'total_ingresos'] = -1
    r = vf.agregar_features_flujo(d)
    assert pd.isna(r.loc[2, 'estancia_promedio_actual'])
    assert pd.isna(r.loc[2, 'balance_flujo_mes'])
    assert not np.isinf(r[vf.FEATURES].to_numpy()).any()


def test_duplicados_colisiones_y_periodo_incoherente_rechazados():
    d = pequeno()
    for malo in [pd.concat([d, d.iloc[:1]], ignore_index=True),
                 d.assign(balance_flujo_mes=1), d.assign(periodo_actual='2020-01')]:
        with pytest.raises(ValueError):
            vf.agregar_features_flujo(malo)


def historico():
    filas = []
    for p in pd.period_range('2015-01', '2026-12', freq='M'):
        for clase, ocupacion in enumerate([.5, .75, .9]):
            filas.append(dict(codigo_ipress=str(clase), servicio_hospitalizacion='S',
                periodo_actual=str(p-1), periodo_predicho=str(p), anio=(p-1).year, mes=(p-1).month,
                ocupacion_estimada=ocupacion, total_pacientes_camas=100., total_camas_disponibles=300.,
                total_ingresos=30., total_egresos=20., total_estancias=50., presion_ingresos_camas=3.,
                nivel_riesgo_actual_codificado=clase, nivel_riesgo_siguiente_mes_codificado=clase))
    return pd.DataFrame(filas)


def test_2025_no_influye_en_desarrollo_y_folds_solo_pasado():
    d = historico()
    h, folds, _ = ev.preparar_desarrollo(d)
    d.loc[d.periodo_predicho.ge('2025-01'), ev.bt.OBJETIVO] = 999
    pd.testing.assert_frame_equal(h, ev.preparar_desarrollo(d)[0])
    assert [f[0] for f in folds] == list(ev.ANIOS)
    for anio, train, test in folds:
        assert h.loc[train, 'periodo_predicho'].lt(f'{anio}-01').all()
        assert h.loc[test, 'periodo_predicho'].str.startswith(str(anio)).all()


def tabla():
    filas = []
    for n in ev.VARIANTES:
        for anio in ev.ANIOS:
            m = dict.fromkeys(ev.bt.METRICAS, 0.)
            m.update(f1_macro=.7, balanced_accuracy=.71, recall_alto=.75,
                     tasa_falsos_negativos_alto=.25, proporcion_alto_bajo=.05)
            filas.append(dict(modelo=n, anio_prueba=anio, fase='desarrollo', tipo='modelo',
                              n_test=30, test_sha256=str(anio), **m))
    return pd.DataFrame(filas)


def test_ranking_f1_restriccion_y_empate_D():
    r = tabla()
    assert ev.seleccionar(r)[1] == 'D'
    r.loc[r.modelo.eq('D+FLUJO'), 'f1_macro'] = .71
    r.loc[r.modelo.eq('D+DINAMICA'), ['balanced_accuracy', 'recall_alto']] = [.99, .99]
    r.loc[r.modelo.eq('D+DEMANDA'), 'f1_macro'] = .6799
    resumen, ganador = ev.seleccionar(r)
    assert ganador == 'D+FLUJO'
    assert not resumen.set_index('modelo').loc['D+DEMANDA', 'admisible']


@pytest.mark.parametrize('caso', ['2025', 'test', 'duplicado', 'fase'])
def test_ranking_rechaza_datos_invalidos(caso):
    r = tabla()
    if caso == '2025':
        r.loc[0, 'anio_prueba'] = 2025
    elif caso == 'test':
        r.loc[0, 'test_sha256'] = 'otro'
    elif caso == 'duplicado':
        r = pd.concat([r, r.iloc[:1]])
    else:
        r.loc[0, 'fase'] = 'comprobacion_2025'
    with pytest.raises(ValueError):
        ev.seleccionar(r)


@pytest.mark.parametrize('mejora', [True, False])
def test_flujo_mismos_tests_produccion_y_seleccion_congelada(tmp_path, monkeypatch, mejora):
    d = historico()
    original = d.copy(deep=True)
    modelo_path = tmp_path/'modelo.joblib'
    modelo_path.write_bytes(b'produccion')
    eventos = []
    class Algoritmo:
        def get_params(self):
            return {'random_state': 42, 'n_estimators': 300}
    class Pipeline:
        classes_ = [0, 1, 2]
        def fit(self, X, y, **kwargs):
            eventos.append(X.copy())
            return self
        def predict(self, X):
            return self.predict_proba(X).argmax(axis=1)
        def predict_proba(self, X):
            c = np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
            if not mejora or 'balance_flujo_mes' not in X:
                c[c == 2] = 0
            return np.eye(3)[c]*.8+.2/3
    motor = SimpleNamespace(COLUMNAS_PREDICTORAS=['ocupacion_estimada'], COLUMNAS_EXCLUIDAS=list(ev.bt.ETIQUETAS),
        obtener_modelos=lambda: {'XGBoost': Algoritmo()}, crear_pipeline=lambda X, alg: Pipeline())
    ajustar = ev.ef._ajustar_evaluar
    def observar(motor, alg, datos, train, test, cols, anio):
        assert not set(cols) & ev.bt.ETIQUETAS
        assert set(cols)-{'ocupacion_estimada', *ev.ef.CONJUNTOS['D']} <= set(vf.FEATURES)
        if anio == 2025:
            j = json.loads((tmp_path/ev.ARCHIVOS[2]).read_text())
            assert j['variante_seleccionada'] == ('D+FLUJO' if mejora else 'D')
            assert j['evaluacion_2025'] is None
        return ajustar(motor, alg, datos, train, test, cols, anio)
    monkeypatch.setattr(ev.ef, '_ajustar_evaluar', observar)
    r, s, j = ev.evaluar(d, tmp_path, motor=motor)
    assert len(r) == (27 if mejora else 26)
    assert [len(v) for v in vf.VARIANTES.values()] == [0, 5, 9, 13, 17]
    for _, g in r.groupby('anio_prueba'):
        assert g.test_sha256.nunique() == g.n_test.nunique() == 1
    assert len(eventos) == len(r)
    assert len(j['definiciones_variables']) == 17
    assert not j['2025_participo_en_seleccion'] and not j['es_modelo_final_produccion']
    assert modelo_path.read_bytes() == b'produccion'
    pd.testing.assert_frame_equal(d, original)
    assert {'recall_alto', 'precision_alto', 'errores_alto_bajo'}.issubset(pd.read_csv(tmp_path/ev.ARCHIVOS[0]))
    with pytest.raises(FileExistsError):
        ev.evaluar(d, tmp_path, motor=motor)
