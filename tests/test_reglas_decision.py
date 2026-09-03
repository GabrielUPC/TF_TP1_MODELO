import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import evaluar_reglas_decision as rd


def datos():
    filas = []
    for periodo in pd.period_range('2015-01', '2026-12', freq='M'):
        actual = periodo-1
        for clase, ocupacion in enumerate([.5, .75, .9]):
            filas.append(dict(codigo_ipress=str(clase), servicio_hospitalizacion='S',
                periodo_actual=str(actual), periodo_predicho=str(periodo), anio=actual.year, mes=actual.month,
                ocupacion_estimada=ocupacion, total_pacientes_camas=100, total_camas_disponibles=300,
                total_ingresos=30, total_egresos=20, presion_ingresos_camas=3.,
                nivel_riesgo_actual_codificado=clase, nivel_riesgo_siguiente_mes_codificado=clase))
    return pd.DataFrame(filas)


def motor_falso():
    eventos = []
    class Algoritmo:
        def get_params(self):
            return {'random_state': 42, 'n_estimators': 300}
    class Pipeline:
        # Orden intencionalmente distinto: el experimento debe reordenar.
        classes_ = [2, 0, 1]
        def fit(self, X, y, **kwargs):
            assert set(X).isdisjoint(rd.bt.ETIQUETAS)
            eventos.append(('fit', X.copy(), kwargs))
            return self
        def predict(self, X):
            raise AssertionError('La referencia debe calcular argmax desde predict_proba compartido.')
        def predict_proba(self, X):
            eventos.append(('proba', X.copy(), {}))
            clases = np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
            canonicas = np.array([[.7, .1, .2], [.2, .5, .3], [.4, .25, .35]])[clases]
            return canonicas[:, [2, 0, 1]]
    return SimpleNamespace(COLUMNAS_PREDICTORAS=['ocupacion_estimada'],
        COLUMNAS_EXCLUIDAS=list(rd.bt.ETIQUETAS), obtener_modelos=lambda: {'XGBoost': Algoritmo()},
        crear_pipeline=lambda X, alg: Pipeline(), eventos=eventos)


def test_espacio_doce_reglas_sin_combinaciones_extra():
    reglas = rd.reglas_candidatas()
    assert len(reglas) == len({r['regla'] for r in reglas}) == 12
    assert reglas[0]['regla'] == rd.BASE
    assert [r['umbral'] for r in reglas if r['tipo_regla'] == 'alto'] == [.25, .30, .35, .40, .45]
    assert [r['umbral'] for r in reglas if r['tipo_regla'] == 'proteccion'] == [.20, .25, .30, .35]
    assert [(r['umbral'], r['umbral_proteccion']) for r in reglas if r['tipo_regla'] == 'combinada'] == [(.35, .25), (.40, .25)]


def test_argmax_empates_y_probabilidades_intactas():
    p = np.array([[.5, .3, .2], [.2, .4, .4], [.1, .2, .7]])
    original = p.copy()
    p.setflags(write=False)
    np.testing.assert_array_equal(rd.aplicar_regla(p, rd.reglas_candidatas()[0]), [0, 1, 2])
    for regla in rd.reglas_candidatas():
        rd.aplicar_regla(p, regla)
    np.testing.assert_array_equal(p, original)


@pytest.mark.parametrize('umbral', [.25, .30, .35, .40, .45])
def test_promocion_alto_inclusiva_y_respaldo_argmax(umbral):
    regla = next(r for r in rd.reglas_candidatas() if r['regla'] == f'alto_{umbral:.2f}')
    p = [[.5, .5-umbral, umbral], [.8, .1, .1]]
    np.testing.assert_array_equal(rd.aplicar_regla(p, regla), [2, 0])


@pytest.mark.parametrize('umbral', [.20, .25, .30, .35])
def test_proteccion_solo_bajo_y_no_mejora_recall_alto(umbral):
    regla = next(r for r in rd.reglas_candidatas() if r['regla'] == f'proteccion_{umbral:.2f}')
    p = np.array([[.5, .5-umbral, umbral], [.1, .8, .1], [.1, .1, .8]])
    base = rd.aplicar_regla(p, rd.reglas_candidatas()[0])
    protegido = rd.aplicar_regla(p, regla)
    np.testing.assert_array_equal(protegido, [1, 1, 2])
    m_base = rd.bt.calcular_metricas_backtesting([2, 1, 2], base)
    m_prot = rd.bt.calcular_metricas_backtesting([2, 1, 2], protegido)
    assert m_prot['errores_alto_bajo'] < m_base['errores_alto_bajo']
    assert m_prot['recall_alto'] == m_base['recall_alto']
    assert m_prot['tasa_falsos_negativos_alto'] == m_base['tasa_falsos_negativos_alto']


@pytest.mark.parametrize('p', [[[.4, .4]], [[np.nan, .5, .5]], [[-.1, .5, .6]], [[.4, .4, .4]], []])
def test_probabilidades_invalidas_se_rechazan(p):
    with pytest.raises(ValueError, match='Probabilidades'):
        rd.aplicar_regla(p, rd.reglas_candidatas()[0])


def resultados():
    filas = []
    for regla in rd.reglas_candidatas():
        for anio in rd.ANIOS_DESARROLLO:
            metricas = dict.fromkeys(rd.bt.METRICAS, 0.)
            metricas.update(f1_macro=.7, balanced_accuracy=.71, recall_alto=.75,
                            tasa_falsos_negativos_alto=.25, proporcion_alto_bajo=.05)
            metricas.update({f'predichos_{c}': 10 for c in ['bajo', 'medio', 'alto']})
            metricas.update({f'proporcion_predicha_{c}': 1/3 for c in ['bajo', 'medio', 'alto']})
            metricas['riesgo_expansion_alto'] = False
            filas.append(dict(regla=regla['regla'], tipo='regla_decision', anio_prueba=anio,
                              n_test=30, test_sha256=str(anio), probabilidades_sha256=str(anio),
                              fase='desarrollo', **metricas))
    return pd.DataFrame(filas)


def test_ranking_prioriza_severos_rechaza_caida_f1_y_prefiere_base_en_empate():
    r = resultados()
    assert rd.seleccionar_regla(r)[1] == rd.BASE
    # Alto->Bajo prima, incluso si otra regla obtiene menor FNR.
    r.loc[r.regla.eq('alto_0.25'), ['proporcion_alto_bajo', 'recall_alto', 'tasa_falsos_negativos_alto']] = [.03, .95, .05]
    r.loc[r.regla.eq('proteccion_0.25'), 'proporcion_alto_bajo'] = .02
    r.loc[r.regla.eq('alto_0.30'), ['proporcion_alto_bajo', 'f1_macro', 'tasa_falsos_negativos_alto', 'recall_alto']] = [0., .6799, .01, .99]
    resumen, elegida = rd.seleccionar_regla(r)
    assert elegida == 'proteccion_0.25'
    assert not resumen.set_index('regla').loc['alto_0.30', 'admisible']
    # Caída exactamente 0.02 sí es admisible.
    r.loc[r.regla.eq('alto_0.30'), 'f1_macro'] = .68
    assert rd.seleccionar_regla(r)[1] == 'alto_0.30'


def test_fnr_mejora_con_igual_proporcion_severa():
    r = resultados()
    r.loc[r.regla.eq('alto_0.40'), ['tasa_falsos_negativos_alto', 'recall_alto']] = [.2, .8]
    assert rd.seleccionar_regla(r)[1] == 'alto_0.40'


def test_severos_desempatan_cuando_fnr_es_igual():
    r = resultados()
    r.loc[r.regla.isin(['alto_0.25', 'alto_0.30']), ['tasa_falsos_negativos_alto', 'recall_alto']] = [.2, .8]
    r.loc[r.regla.eq('alto_0.30'), 'proporcion_alto_bajo'] = .01
    assert rd.seleccionar_regla(r)[1] == 'alto_0.30'


@pytest.mark.parametrize('caso', ['2025', 'sin_argmax', 'otro_test', 'otra_proba', 'duplicado'])
def test_ranking_rechaza_comparaciones_invalidas(caso):
    r = resultados()
    if caso == '2025':
        r.loc[r.anio_prueba.eq(2024), 'anio_prueba'] = 2025
    elif caso == 'sin_argmax':
        r = r[r.regla.ne(rd.BASE)]
    elif caso == 'otro_test':
        r.loc[0, 'test_sha256'] = 'otro'
    elif caso == 'otra_proba':
        r.loc[0, 'probabilidades_sha256'] = 'otra'
    else:
        r = pd.concat([r, r.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        rd.seleccionar_regla(r)


def test_folds_no_consultan_etiquetas_2025_y_no_usan_futuro():
    df = datos()
    historico, folds, _ = rd.ef.preparar_desarrollo(df)
    cambiado = df.copy()
    cambiado.loc[cambiado.periodo_predicho.ge('2025-01'), rd.bt.OBJETIVO] = 999
    pd.testing.assert_frame_equal(historico, rd.ef.preparar_desarrollo(cambiado)[0])
    for anio, train, test in folds:
        assert historico.loc[train, 'periodo_predicho'].lt(f'{anio}-01').all()
        assert historico.loc[test, 'periodo_predicho'].str.startswith(str(anio)).all()


def test_flujo_un_fit_por_anio_mismas_proba_2025_congelado_y_produccion(tmp_path, monkeypatch):
    df = datos()
    original = df.copy(deep=True)
    motor = motor_falso()
    sentinel = tmp_path/'modelo_ipress.joblib'
    sentinel.write_bytes(b'produccion sin cambios')
    funcion = rd.probabilidades_fold
    def observar(motor, algoritmo, datos_fold, train, test, columnas, anio):
        assert set(columnas) == {'ocupacion_estimada', *rd.CONJUNTOS['D']}
        if anio == 2025:
            j = json.loads((tmp_path/rd.ARCHIVOS[2]).read_text())
            assert j['regla_seleccionada']['regla'] == 'alto_0.35'
            assert j['2025_participo_en_seleccion'] is False
            assert j['evaluacion_2025'] is None
        return funcion(motor, algoritmo, datos_fold, train, test, columnas, anio)
    monkeypatch.setattr(rd, 'probabilidades_fold', observar)
    res, resumen, reporte = rd.evaluar_reglas(df, tmp_path, motor=motor)
    assert len(res) == 62
    assert len(resumen) == 12
    assert reporte['regla_seleccionada']['regla'] == 'alto_0.35'
    assert reporte['es_modelo_final_produccion'] is False
    assert reporte['evaluacion_2025']['n_ajustes_XGBoost_D'] == 1
    assert len([e for e, _, _ in motor.eventos if e == 'fit']) == 6
    assert len([e for e, _, _ in motor.eventos if e == 'proba']) == 6
    for fold, anio in enumerate([*rd.ANIOS_DESARROLLO, 2025]):
        entrenamiento, prueba = motor.eventos[fold*2:fold*2+2]
        assert df.loc[entrenamiento[1].index, 'periodo_predicho'].lt(f'{anio}-01').all()
        assert df.loc[prueba[1].index, 'periodo_predicho'].str.startswith(str(anio)).all()
    for _, grupo in res.groupby('anio_prueba'):
        assert grupo.test_sha256.nunique() == grupo.probabilidades_sha256.nunique() == 1
        # AUC no cambia: las probabilidades son idénticas, solo cambia decisión.
        assert grupo.roc_auc_ovr_macro.nunique() == 1
    assert {'regla', 'fase', 'precision_alto', 'errores_alto_bajo'}.issubset(pd.read_csv(tmp_path/rd.ARCHIVOS[0]))
    assert {'ranking', 'admisible', 'seleccionada'}.issubset(pd.read_csv(tmp_path/rd.ARCHIVOS[1]))
    j = json.loads((tmp_path/rd.ARCHIVOS[2]).read_text())
    assert {'reglas_probadas', 'anios_desarrollo', 'criterio_seleccion', 'metricas_promedio',
            'comparacion_argmax', 'evaluacion_2025'}.issubset(j)
    assert j['regla_seleccionada']['regla'] == reporte['regla_seleccionada']['regla']
    assert len(j['calibracion_por_anio']) == 5
    assert j['observaciones_calibracion']['calibrador_ajustado'] is False
    calibracion = pd.read_csv(tmp_path/rd.ARCHIVOS[3])
    assert len(calibracion) == 150
    assert set(calibracion.anio_prueba) == set(rd.ANIOS_DESARROLLO)
    assert {'log_loss', 'brier_multiclase', 'confianza_media_maxima', 'ece_clase', 'n_bin'}.issubset(calibracion)
    assert res[['predichos_bajo', 'predichos_medio', 'predichos_alto']].sum(axis=1).eq(res.n_test).all()
    assert sentinel.read_bytes() == b'produccion sin cambios'
    assert motor.COLUMNAS_PREDICTORAS == ['ocupacion_estimada']
    pd.testing.assert_frame_equal(df, original)
    with pytest.raises(FileExistsError):
        rd.evaluar_reglas(df, tmp_path, motor=motor)
    assert len(motor.eventos) == 12


@pytest.mark.parametrize('umbral', [.35, .40])
def test_combinadas_prioridad_alto_y_bordes_deterministas(umbral):
    regla = next(r for r in rd.reglas_candidatas() if r['regla'] == f'combinada_{umbral:.2f}_0.25')
    p = np.array([[.5, .5-umbral, umbral], [.5, .25, .25], [.6, .2, .2], [.1, .65, .25]])
    copia = p.copy()
    np.testing.assert_array_equal(rd.aplicar_regla(p, regla), [2, 1, 0, 1])
    np.testing.assert_array_equal(rd.aplicar_regla(p, regla), rd.aplicar_regla(p, regla))
    np.testing.assert_array_equal(p, copia)


def test_calibracion_formulas_probabilidades_intactas_bins_y_no_2025():
    y = [0, 1, 2]
    p = np.full((3, 3), 1/3)
    original = p.copy()
    r, tabla = rd.diagnosticar_calibracion(y, p, 2024)
    assert r['log_loss'] == pytest.approx(np.log(3))
    assert r['brier_multiclase'] == pytest.approx(2/3)
    assert r['confianza_media_maxima'] == pytest.approx(1/3)
    assert r['ece_macro'] == pytest.approx(0.)
    assert len(tabla) == 30
    assert tabla.groupby('clase').n_bin.sum().eq(3).all()
    assert tabla.loc[tabla.n_bin.eq(0), 'frecuencia_observada'].isna().all()
    np.testing.assert_array_equal(p, original)
    perfecto, bins = rd.diagnosticar_calibracion(y, np.eye(3), 2018)
    assert perfecto['brier_multiclase'] == perfecto['log_loss'] == 0
    assert bins.loc[bins.bin.eq(9)].n_bin.eq(1).all()
    with pytest.raises(ValueError, match='desarrollo'):
        rd.diagnosticar_calibracion(y, p, 2025)


def test_calibracion_mala_solo_alerta_no_recalibra():
    p = np.tile([1., 0., 0.], (120, 1))
    r, _ = rd.diagnosticar_calibracion(np.ones(120), p, 2023)
    assert r['alerta_calibracion'] is True
    assert r['brier_multiclase'] == 2
    assert np.isfinite(r['log_loss'])
    np.testing.assert_array_equal(p, np.tile([1., 0., 0.], (120, 1)))


def test_mejora_minima_no_compensa_caida_global_y_expansion_es_alerta():
    r = resultados()
    mascara = r.regla.eq('alto_0.25')
    r.loc[mascara, ['proporcion_alto_bajo', 'f1_macro']] = [.049, .689]
    r.loc[mascara, 'proporcion_predicha_alto'] = .6
    r.loc[mascara, 'riesgo_expansion_alto'] = True
    s, elegida = rd.seleccionar_regla(r)
    fila = s.set_index('regla').loc['alto_0.25']
    assert fila.admisible and fila.veto_mejora_minima_caida_global
    assert not fila.elegible_seleccion and elegida == 'argmax'
    assert fila.riesgo_expansion_alto_promedio and fila.riesgo_expansion_alto_algun_anio
    # Mejora suficiente mantiene elegibilidad, aun con alerta de expansión.
    r.loc[mascara, 'proporcion_alto_bajo'] = .04
    assert rd.seleccionar_regla(r)[1] == 'alto_0.25'


def test_calibracion_se_diagnostica_antes_de_aplicar_reglas(tmp_path, monkeypatch):
    eventos = []
    original_diag = rd.diagnosticar_calibracion
    original_eval = rd.evaluar_probabilidades
    def diagnostico(y, p, anio):
        eventos.append(anio)
        return original_diag(y, p, anio)
    def evaluar(df, train, test, p, anio, reglas, fase):
        if fase == 'desarrollo':
            assert anio in eventos
        return original_eval(df, train, test, p, anio, reglas, fase)
    monkeypatch.setattr(rd, 'diagnosticar_calibracion', diagnostico)
    monkeypatch.setattr(rd, 'evaluar_probabilidades', evaluar)
    rd.evaluar_reglas(datos(), tmp_path, motor=motor_falso())
    assert eventos == list(rd.ANIOS_DESARROLLO)


@pytest.mark.parametrize('umbral', [.40, .35])
def test_extension_020_seis_reglas_limites_prioridad_y_probabilidades_intactas(umbral):
    reglas = rd.reglas_extension_020()
    assert [r['regla'] for r in reglas] == ['argmax', 'proteccion_0.20', 'alto_0.40', 'alto_0.35',
        'combinada_0.40_0.20', 'combinada_0.35_0.20']
    assert len(rd.reglas_candidatas()) == 12
    regla = next(r for r in reglas if r['regla'] == f'combinada_{umbral:.2f}_0.20')
    p = np.array([[.5, .5-umbral, umbral], [.6, .2, .2], [.7, .11, .19], [.1, .7, .2]])
    original = p.copy()
    np.testing.assert_array_equal(rd.aplicar_regla(p, regla), [2, 1, 0, 1])
    np.testing.assert_array_equal(p, original)


def test_extension_reutiliza_hashes_sin_entrenar_ni_consultar_2025(monkeypatch):
    df = datos()
    historico, folds, _ = rd.ef.preparar_desarrollo(df)
    matrices, referencias = {}, []
    for anio, train, test in folds:
        p = np.tile([.3, .3, .4], (len(test), 1))
        matrices[anio] = p
        referencias.append(rd.evaluar_probabilidades(historico, train, test, p, anio,
            [rd.reglas_candidatas()[0]], 'desarrollo'))
    ref = pd.concat(referencias, ignore_index=True)
    def prohibido(*args, **kwargs):
        raise AssertionError('La extensión no debe ajustar ni pedir probabilidades nuevas.')
    monkeypatch.setattr(rd, 'probabilidades_fold', prohibido)
    monkeypatch.setattr(rd.bt, '_motor_existente', prohibido)
    original = df.copy(deep=True)
    df.loc[df.periodo_predicho.ge('2025-01'), rd.bt.OBJETIVO] = 999
    r, s, elegida = rd.evaluar_extension_020(df, matrices, ref)
    assert len(r) == 30 and len(s) == 6
    assert set(r.anio_prueba) == set(rd.ANIOS_DESARROLLO)
    assert 'reduce_ambos_errores' in s
    assert r[['predichos_bajo', 'predichos_medio', 'predichos_alto']].sum(axis=1).eq(r.n_test).all()
    with pytest.raises(ValueError, match='no 2025'):
        rd.evaluar_extension_020(df, {**matrices, 2025: matrices[2018]}, ref)
    adulteradas = {**matrices, 2018: matrices[2018][:, [2, 1, 0]].copy()}
    with pytest.raises(ValueError, match='no coinciden'):
        rd.evaluar_extension_020(df, adulteradas, ref)
    final = original.loc[original.periodo_predicho.lt('2026-01')]
    f2025 = next(f for f in rd.bt.crear_folds_expansivos(final)[0] if f[0] == 2025)
    anio, train, test = f2025
    p = np.tile([.3, .3, .4], (len(test), 1))
    ref2025 = rd.evaluar_probabilidades(final, train, test, p, 2025, [rd.reglas_candidatas()[0]], 'comprobacion_2025')
    comprobacion = rd.comprobar_extension_020_2025(original, p, ref2025, elegida)
    assert set(comprobacion.regla) == {rd.BASE, elegida}
