import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import evaluar_dataset_a as ea


def datos():
    filas = []
    for periodo in pd.period_range('2015-01', '2026-12', freq='M'):
        actual = periodo-1
        for clase, ocupacion in enumerate([.5, .75, .9]):
            filas.append(dict(codigo_ipress=str(clase).zfill(8), servicio_hospitalizacion='S',
                periodo_actual=str(actual), periodo_predicho=str(periodo), anio=actual.year, mes=actual.month,
                ocupacion_estimada=ocupacion, total_pacientes_camas=100, total_camas_disponibles=300,
                total_camas=10, total_ingresos=30, total_egresos=20, presion_ingresos_camas=3.,
                nivel_riesgo_actual_codificado=clase, nivel_riesgo_siguiente_mes_codificado=clase))
    d = pd.DataFrame(filas)
    a = d[ea.pa.CLAVE].copy()
    for v in ea.pa.VARIABLES:
        a[v] = 10.
    a['estado_calidad_a'] = 'UNICA'
    a = ea.pa.agregar_derivadas(a)
    # Un mes faltante: no se descarta el D correspondiente.
    a = a.iloc[1:].copy()
    a = ea.pa.agregar_derivadas(a)
    return d, a, ea.pa.unir_d_a(d, a)


def motor_falso(mejora=True):
    eventos = []
    class Algoritmo:
        def get_params(self):
            return {'random_state': 42, 'n_estimators': 300}
    class Pipeline:
        classes_ = [0, 1, 2]
        def fit(self, X, y, **kwargs):
            eventos.append(('train', X.copy(), kwargs))
            return self
        def predict(self, X):
            eventos.append(('test', X.copy(), {}))
            return self.predict_proba(X).argmax(axis=1)
        def predict_proba(self, X):
            c = np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
            if not mejora or 'CA_CAMAS' not in X:
                c[c == 2] = 0
            return np.eye(3)[c]*.8+.2/3
    return SimpleNamespace(COLUMNAS_PREDICTORAS=['ocupacion_estimada'],
        COLUMNAS_EXCLUIDAS=list(ea.bt.ETIQUETAS), obtener_modelos=lambda: {'XGBoost': Algoritmo()},
        crear_pipeline=lambda X, a: Pipeline(), eventos=eventos)


def resultados():
    filas = []
    for nombre in ea.VARIANTES:
        for anio in ea.ANIOS:
            m = dict.fromkeys(ea.bt.METRICAS, 0.)
            m.update(f1_macro=.7, balanced_accuracy=.71, recall_alto=.75,
                     tasa_falsos_negativos_alto=.25, proporcion_alto_bajo=.05)
            filas.append(dict(modelo=nombre, anio_prueba=anio, fase='desarrollo', tipo='modelo',
                              n_test=30, test_sha256=str(anio), **m))
    return pd.DataFrame(filas)


def test_cinco_variantes_exactas_sin_indicador_extra():
    assert [len(v) for v in ea.VARIANTES.values()] == [0, 1, 4, 7, 10]
    assert all('tiene_datos_a' not in v for v in ea.VARIANTES.values())


def test_ranking_f1_primero_limite_caida_y_desempate_D():
    r = resultados()
    assert ea.seleccionar(r)[1] == 'D'
    r.loc[r.modelo.eq('D+A_CAMAS'), 'f1_macro'] = .71
    r.loc[r.modelo.eq('D+A_COMPLETO'), ['balanced_accuracy', 'recall_alto']] = [.99, .99]
    assert ea.seleccionar(r)[1] == 'D+A_CAMAS'
    r.loc[r.modelo.eq('D+A_RECURSOS'), 'f1_macro'] = .6799
    assert not ea.seleccionar(r)[0].set_index('modelo').loc['D+A_RECURSOS', 'admisible']


@pytest.mark.parametrize('campo,valor', [('balanced_accuracy', .8), ('recall_alto', .8),
    ('tasa_falsos_negativos_alto', .2), ('proporcion_alto_bajo', .01)])
def test_desempates(campo, valor):
    r = resultados()
    r.loc[r.modelo.eq('D+A_RECURSOS'), campo] = valor
    assert ea.seleccionar(r)[1] == 'D+A_RECURSOS'


@pytest.mark.parametrize('caso', ['2025', 'duplicado', 'test', 'fase', 'sin_D'])
def test_rechaza_seleccion_invalida(caso):
    r = resultados()
    if caso == '2025':
        r.loc[r.anio_prueba.eq(2024), 'anio_prueba'] = 2025
    elif caso == 'duplicado':
        r = pd.concat([r, r.iloc[:1]])
    elif caso == 'test':
        r.loc[0, 'test_sha256'] = 'otro'
    elif caso == 'fase':
        r.loc[0, 'fase'] = 'ablacion'
    else:
        r = r.loc[r.modelo.ne('D')]
    with pytest.raises(ValueError):
        ea.seleccionar(r)


def test_validacion_join_ambigua_nan_y_t_mas_uno():
    d, a, x = datos()
    vacia = a[ea.pa.CLAVE].iloc[:0]
    ea.validar_candidato(d, a, x, vacia)
    assert x.loc[x.tiene_datos_a.eq(0), ea.pa.CANDIDATAS].isna().all().all()
    assert len(x) == len(d)
    with pytest.raises(ValueError, match='ambigua'):
        ea.validar_candidato(d, a, x, a[ea.pa.CLAVE].iloc[:1])
    with pytest.raises(ValueError, match='filas'):
        ea.validar_candidato(d, a, pd.concat([x, x.iloc[:1]]), vacia)
    malo = x.copy()
    malo.loc[0, 'CA_CAMAS'] = 10  # rellenar con el siguiente mes está prohibido
    with pytest.raises(AssertionError):
        ea.validar_candidato(d, a, malo, vacia)
    malo = a.copy()
    malo.loc[malo.index[0], 'variacion_camas_a_1m'] = 0
    with pytest.raises(AssertionError):
        ea.validar_candidato(d, malo, x, vacia)


def test_2025_no_afecta_desarrollo_y_temporales_a_solo_pasado():
    _, a, x = datos()
    h, folds, _ = ea.ef.preparar_desarrollo(x)
    cambiado = x.copy()
    cambiado.loc[cambiado.periodo_predicho.ge('2025-01'), ea.bt.OBJETIVO] = 999
    pd.testing.assert_frame_equal(h, ea.ef.preparar_desarrollo(cambiado)[0])
    for anio, train, test in folds:
        assert h.loc[train, 'periodo_predicho'].lt(f'{anio}-01').all()
        assert h.loc[test, 'periodo_predicho'].str.startswith(str(anio)).all()
    futuro = a.copy()
    futuro.loc[futuro.anio.ge(2025), ea.pa.VARIABLES] = 999
    futuro = ea.pa.agregar_derivadas(futuro)
    pd.testing.assert_frame_equal(a.loc[a.anio.lt(2025)].reset_index(drop=True),
                                 futuro.loc[futuro.anio.lt(2025)].reset_index(drop=True))


def test_flujo_mismos_tests_nan_ablacion_congelada_y_no_produccion(tmp_path, monkeypatch):
    _, _, x = datos()
    original = x.copy(deep=True)
    motor = motor_falso()
    protegido = tmp_path/'modelo.joblib'
    protegido.write_bytes(b'produccion')
    original_fit = ea.ef._ajustar_evaluar
    def observar(motor, algoritmo, df, train, test, columnas, anio):
        if anio == 2025:
            j = json.loads((tmp_path/ea.ARCHIVOS[2]).read_text())
            assert j['variante_seleccionada'] == 'D+A_CAMAS'
            assert j['2025_participo_en_seleccion'] is False
            assert j['evaluacion_2025'] is None
        return original_fit(motor, algoritmo, df, train, test, columnas, anio)
    monkeypatch.setattr(ea.ef, '_ajustar_evaluar', observar)
    r, s, j = ea.evaluar(x, tmp_path, motor=motor)
    assert j['variante_seleccionada'] == 'D+A_CAMAS'
    assert len(r) == 25+5+2
    assert len(j['ablacion']) == 5
    assert set(r.loc[r.fase.eq('ablacion'), 'variable_retirada']) == {'CA_CAMAS'}
    assert j['ablacion_modifica_seleccion'] is False
    assert len(j['evaluacion_2025']) == 2
    for _, g in r.groupby('anio_prueba'):
        assert g.test_sha256.nunique() == 1 and g.n_test.nunique() == 1
    for tipo, X, _ in motor.eventos:
        if 'CA_CAMAS' in X:
            np.testing.assert_allclose(X.CA_CAMAS, x.loc[X.index, 'CA_CAMAS'], equal_nan=True)
        assert set(X).isdisjoint(ea.bt.ETIQUETAS)
    assert protegido.read_bytes() == b'produccion'
    pd.testing.assert_frame_equal(x, original)
    assert j['es_modelo_final_produccion'] is False
    assert set(ea.ARCHIVOS).issubset(p.name for p in tmp_path.iterdir())
    assert {'fase', 'recall_alto', 'roc_auc_ovr_macro'}.issubset(pd.read_csv(tmp_path/ea.ARCHIVOS[0]))
    assert {'ranking', 'delta_f1_macro_promedio'}.issubset(pd.read_csv(tmp_path/ea.ARCHIVOS[1]))
    with pytest.raises(FileExistsError):
        ea.evaluar(x, tmp_path, motor=motor)


def test_si_gana_D_no_hay_ablacion_ni_doble_evaluacion_2025(tmp_path):
    _, _, x = datos()
    r, _, j = ea.evaluar(x, tmp_path, motor=motor_falso(mejora=False))
    assert j['variante_seleccionada'] == 'D'
    assert j['ablacion'] == []
    assert len(j['evaluacion_2025']) == 1
    assert len(r) == 26
