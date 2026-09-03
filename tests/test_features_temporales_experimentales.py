import ast
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import variables_temporales_experimentales as vt
from src import evaluar_features_temporales as ef


def datos(inicio='2015-01', fin='2026-12'):
    filas = []
    for periodo in pd.period_range(inicio, fin, freq='M'):
        actual = periodo - 1
        for clase in range(3):
            filas.append(dict(codigo_ipress=str(clase), servicio_hospitalizacion='S',
                anio=actual.year, mes=actual.month, periodo_actual=str(actual), periodo_predicho=str(periodo),
                ocupacion_estimada=[.5, .75, .9][clase], total_pacientes_camas=100+actual.month,
                total_camas_disponibles=300+actual.month, total_ingresos=30+actual.month,
                total_egresos=20+actual.month, ratio_camas_disponibles=1., distrito="D",
                presion_ingresos_camas=actual.month/10, nivel_riesgo_actual_codificado=clase,
                nivel_riesgo_siguiente_mes_codificado=clase))
    return pd.DataFrame(filas)


def historia():
    df = datos('2023-02', '2023-05')
    df = df.loc[df.codigo_ipress.eq('0')].copy()
    df['ocupacion_estimada'] = [.6, .8, .9, .4]
    df['nivel_riesgo_actual_codificado'] = [0, 1, 2, 0]
    return df


def test_lags_cambios_historia_riesgo_y_margenes_exactos():
    df = historia()
    original = df.copy(deep=True)
    r = vt.agregar_features_candidatas(df)
    fila = r.iloc[2]
    assert fila.ocupacion_lag_1m == .8
    assert fila.ocupacion_lag_2m == .6
    assert fila.aceleracion_ocupacion == pytest.approx(-.1)
    assert fila.cambio_ocupacion_2m == pytest.approx(.3)
    assert fila.cambio_pacientes_camas_1m == 1
    assert fila.cambio_camas_disponibles_1m == 1
    assert fila.egresos_lag_2m == 21
    assert r.iloc[3].ocupacion_lag_3m == .6
    assert fila.max_ocupacion_3m == .9
    assert fila.min_ocupacion_3m == .6
    assert fila.desviacion_ocupacion_3m == pytest.approx(np.std([.6, .8, .9]))
    assert fila.max_presion_3m == .3
    assert fila.meses_alto_ultimos_3m == 1
    assert fila.meses_medio_alto_ultimos_3m == 2
    assert fila.margen_umbral_alto == pytest.approx(.05)
    assert fila.margen_umbral_medio == pytest.approx(.2)
    assert r.iloc[0][list(vt.LAGS)].isna().all()
    assert r.iloc[:2].max_ocupacion_3m.isna().all()
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(r[df.columns], df)


def test_modificar_futuro_no_cambia_features_del_pasado_ni_presente():
    df = historia()
    cambiado = df.copy()
    for c in ['ocupacion_estimada', 'total_pacientes_camas', 'total_camas_disponibles',
              'total_ingresos', 'total_egresos', 'presion_ingresos_camas']:
        cambiado.loc[cambiado.index[-1], c] = 9999
    cambiado.loc[cambiado.index[-1], 'nivel_riesgo_actual_codificado'] = 2
    # Incluso etiquetas futuras corruptas son irrelevantes para el generador.
    cambiado[ef.bt.OBJETIVO] = -999
    r1, r2 = vt.agregar_features_candidatas(df), vt.agregar_features_candidatas(cambiado)
    pd.testing.assert_frame_equal(r1.iloc[:-1][vt.NUEVAS_FEATURES], r2.iloc[:-1][vt.NUEVAS_FEATURES])
    sin_objetivo = vt.agregar_features_candidatas(df.drop(columns=ef.bt.OBJETIVO))
    pd.testing.assert_frame_equal(r1[vt.NUEVAS_FEATURES], sin_objetivo[vt.NUEVAS_FEATURES])


def test_hueco_no_saltea_mes_ni_conecta_lag_dos():
    df = historia().drop(index=historia().index[1])  # enero, marzo, abril
    r = vt.agregar_features_candidatas(df)
    assert r.iloc[1][list(vt.LAGS)].isna().all()
    assert pd.isna(r.iloc[1].ocupacion_lag_2m)  # enero no cruza febrero pendiente
    assert pd.isna(r.iloc[2].ocupacion_lag_2m)
    assert r.iloc[2].ocupacion_lag_1m == .9
    assert r.max_ocupacion_3m.isna().all()
    assert r.meses_alto_ultimos_3m.isna().all()


def test_grupos_aislados_orden_original_y_cambio_de_anio():
    df = datos('2023-12', '2024-03').sample(frac=1, random_state=42)
    otro_servicio = df.loc[df.codigo_ipress.eq('0')].assign(servicio_hospitalizacion='OTRO', ocupacion_estimada=.1)
    df = pd.concat([df, otro_servicio], ignore_index=True).sample(frac=1, random_state=7)
    r = vt.agregar_features_candidatas(df)
    assert r.index.equals(df.index)
    enero = r.loc[r.periodo_actual.eq('2024-01')]
    np.testing.assert_allclose(enero.ocupacion_lag_1m, enero.ocupacion_estimada)
    assert enero.max_ocupacion_3m.notna().all()


def test_ausentes_no_se_rellenan_y_duplicados_se_rechazan():
    df = historia()
    df.loc[df.index[1], 'ocupacion_estimada'] = np.inf
    df.loc[df.index[1], 'nivel_riesgo_actual_codificado'] = np.nan
    r = vt.agregar_features_candidatas(df)
    assert pd.isna(r.iloc[2].ocupacion_lag_1m)
    assert pd.isna(r.iloc[2].max_ocupacion_3m)
    assert pd.isna(r.iloc[2].meses_alto_ultimos_3m)
    with pytest.raises(ValueError, match='duplicado'):
        vt.agregar_features_candidatas(pd.concat([historia(), historia()], ignore_index=True))


def test_no_hay_shift_negativo_rolling_centrado_ni_objetivos_como_features():
    assert set(vt.NUEVAS_FEATURES).isdisjoint(ef.bt.ETIQUETAS)
    assert [len(v) for v in vt.CONJUNTOS.values()] == [0, 11, 15, 22, 32]
    arbol = ast.parse(inspect.getsource(vt))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute):
            if nodo.func.attr == 'shift':
                assert all(not isinstance(arg, ast.UnaryOp) for arg in nodo.args)
            if nodo.func.attr == 'rolling':
                assert not any(k.arg == 'center' and isinstance(k.value, ast.Constant) and k.value.value
                               for k in nodo.keywords)
    assert ef.bt.OBJETIVO not in inspect.getsource(vt)


def test_desarrollo_excluye_2025_y_futuro_y_folds_son_expansivos():
    df = datos()
    historico, folds, _ = ef.preparar_desarrollo(df)
    assert tuple(f[0] for f in folds) == ef.ANIOS_DESARROLLO
    anterior = set()
    for anio, train, test in folds:
        assert historico.loc[train, 'periodo_predicho'].lt(f'{anio}-01').all()
        assert historico.loc[test, 'periodo_predicho'].str.startswith(str(anio)).all()
        assert anterior.issubset(set(train))
        anterior = set(train)
    cambiado = df.copy()
    mascara = cambiado.periodo_predicho.ge('2025-01')
    cambiado.loc[mascara, ef.bt.OBJETIVO] = 99
    cambiado.loc[mascara, 'ocupacion_estimada'] = 1e9
    pd.testing.assert_frame_equal(historico, ef.preparar_desarrollo(cambiado)[0])


def resultados():
    filas = []
    for nombre, f1, bal, recall in [('Base', .70, .71, .75), ('A', .71, .72, .76),
                                   ('B', .675, .72, .90), ('C', .685, .70, .80), ('D', .70, .71, .75)]:
        for anio in ef.ANIOS_DESARROLLO:
            metricas = dict.fromkeys(ef.bt.METRICAS, 0.)
            metricas.update(f1_macro=f1, balanced_accuracy=bal, recall_alto=recall,
                            tasa_falsos_negativos_alto=1-recall)
            filas.append(dict(modelo=nombre, tipo='modelo', anio_prueba=anio,
                              test_sha256=str(anio), n_test=30, **metricas))
    return pd.DataFrame(filas)


def test_seleccion_respeta_mejoras_y_limite_f1():
    r = resultados()
    resumen, elegido = ef.seleccionar_conjunto(r)
    assert elegido == 'A'
    tabla = resumen.set_index('modelo')
    assert not tabla.loc['B', 'admisible']
    assert tabla.loc['C', 'admisible']
    r.loc[r.modelo.eq('A'), ['f1_macro', 'balanced_accuracy', 'recall_alto', 'tasa_falsos_negativos_alto']] = [.69, .70, .75, .25]
    assert ef.seleccionar_conjunto(r)[1] == 'C'
    r.loc[r.modelo.eq('C'), ['recall_alto', 'tasa_falsos_negativos_alto']] = [.75, .25]
    assert ef.seleccionar_conjunto(r)[1] == 'Base'


@pytest.mark.parametrize('caso', ['2025', 'sin_base', 'distinto_test'])
def test_seleccion_rechaza_comparacion_invalida(caso):
    r = resultados()
    if caso == '2025':
        r.loc[r.anio_prueba.eq(2024), 'anio_prueba'] = 2025
    elif caso == 'sin_base':
        r = r[r.modelo.ne('Base')]
    else:
        r.loc[0, 'test_sha256'] = 'diferente'
    with pytest.raises(ValueError):
        ef.seleccionar_conjunto(r)


def motor_falso():
    llamadas = []
    class Algoritmo:
        def get_params(self):
            return {'random_state': 42}
    class Pipeline:
        classes_ = [0, 1, 2]
        def fit(self, X, y, **kwargs):
            assert set(X).isdisjoint(ef.bt.ETIQUETAS)
            llamadas.append(('fit', X.copy(), kwargs))
            return self
        def predict(self, X):
            llamadas.append(('predict', X.copy(), {}))
            return np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
        def predict_proba(self, X):
            return np.eye(3)[np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)]
    return SimpleNamespace(COLUMNAS_PREDICTORAS=['ocupacion_estimada', 'ratio_camas_disponibles', 'presion_ingresos_camas', 'anio', 'distrito'], COLUMNAS_EXCLUIDAS=list(ef.bt.ETIQUETAS),
        obtener_modelos=lambda: {'XGBoost': Algoritmo()}, crear_pipeline=lambda X, a: Pipeline(), llamadas=llamadas)


def test_experimento_exporta_sin_tocar_produccion_y_mismo_test(tmp_path):
    df = datos()
    original = df.copy(deep=True)
    sentinel = tmp_path / 'modelo_ipress.joblib'
    sentinel.write_bytes(b'produccion intocable')
    motor = motor_falso()
    resultados, resumen, reporte = ef.evaluar_features(df, tmp_path, motor=motor)
    assert len(resultados) == 47  # 25 desarrollo + 20 ablación + 2 comprobaciones
    assert len(resumen) == 5
    assert reporte['conjunto_seleccionado'] == 'Base'
    assert motor.COLUMNAS_PREDICTORAS == ['ocupacion_estimada', 'ratio_camas_disponibles', 'presion_ingresos_camas', 'anio', 'distrito']
    for anio, grupo in resultados.groupby('anio_prueba'):
        assert grupo.test_sha256.nunique() == 1
        assert grupo.n_test.nunique() == 1
    for fold, anio in enumerate(ef.ANIOS_DESARROLLO):
        for evento, X, kwargs in motor.llamadas[fold*10:(fold+1)*10]:
            periodos = df.loc[X.index, 'periodo_predicho']
            if evento == 'fit':
                assert periodos.lt(f'{anio}-01').all()
                assert len(kwargs['modelo__sample_weight']) == len(X)
            else:
                assert periodos.str.startswith(str(anio)).all()
    assert sentinel.read_bytes() == b'produccion intocable'
    assert {'modelo', 'recall_alto', 'roc_auc_ovr_macro', 'errores_alto_bajo'}.issubset(pd.read_csv(tmp_path/ef.ARCHIVOS[0]))
    assert {'modelo', 'admisible', 'seleccionado'}.issubset(pd.read_csv(tmp_path/ef.ARCHIVOS[1]))
    j = json.loads((tmp_path/ef.ARCHIVOS[2]).read_text(encoding='utf-8'))
    assert j['features_anadidas_por_conjunto'] == vt.CONJUNTOS
    assert j['anios_desarrollo'] == list(ef.ANIOS_DESARROLLO)
    assert j['2025_participo_en_seleccion'] is False
    assert j['es_modelo_final_produccion'] is False
    assert j['2025_evaluado_en_este_experimento'] is True
    assert j['ablacion_cambia_seleccion'] is False
    assert len(j['resultados_ablacion']) == 20
    assert set(j['definiciones_variables']) == set(vt.NUEVAS_FEATURES)
    importancia = pd.read_csv(tmp_path/ef.ARCHIVOS[3])
    assert set(importancia.anio_prueba) == set(ef.ANIOS_DESARROLLO)
    assert {'variable', 'importancia_f1_macro_media', 'n_muestra'}.issubset(importancia)
    test2025 = df.index[df.periodo_predicho.str.startswith('2025')]
    assert sum(e == 'predict' and X.index.equals(test2025) for e, X, _ in motor.llamadas) == 1
    for evento, X, _ in motor.llamadas:
        if evento == 'fit':
            assert df.loc[X.index, 'periodo_predicho'].lt('2025-01').all()
    with pytest.raises(FileExistsError):
        ef.evaluar_features(df, tmp_path, motor=motor)
    pd.testing.assert_frame_equal(df, original)


def test_lag_tres_y_ventanas_seis_no_cruzan_huecos():
    df = datos('2023-02', '2023-08').query('codigo_ipress == "2"').copy()
    df['ocupacion_estimada'] = [.1, .2, .3, .4, .5, .6, .7]
    r = vt.agregar_features_candidatas(df)
    assert r.iloc[3].ocupacion_lag_3m == .1
    assert r.iloc[5].promedio_ocupacion_6m == pytest.approx(.35)
    assert r.iloc[5].meses_alto_ultimos_6m == 6
    assert r.iloc[:5].promedio_ocupacion_6m.isna().all()
    hueco = vt.agregar_features_candidatas(df.drop(index=df.index[1]))
    assert pd.isna(hueco.iloc[2].ocupacion_lag_3m)
    assert hueco.promedio_ocupacion_6m.isna().all()
    # Truncar datos en t no altera ninguna de sus features (incluye aceleración/rachas).
    for n in range(1, len(df)+1):
        parcial = vt.agregar_features_candidatas(df.iloc[:n])
        pd.testing.assert_frame_equal(parcial[vt.NUEVAS_FEATURES], r.iloc[:n][vt.NUEVAS_FEATURES])


def test_racha_alto_se_corta_por_no_alto_ausente_y_hueco():
    df = datos('2023-02', '2023-09').query('codigo_ipress == "2"').copy()
    df['nivel_riesgo_actual_codificado'] = [2, 2, 1, 2, np.nan, 2, 2, 2]
    df = df.drop(index=df.index[-2])
    r = vt.agregar_features_candidatas(df)
    np.testing.assert_allclose(r.meses_consecutivos_alto, [1, 2, 0, 1, np.nan, 1, 1], equal_nan=True)


def test_crecimiento_formula_y_cero_sin_inf():
    df = historia()
    df['total_pacientes_camas'] = [100, 120, 0, 30]
    df['total_camas_disponibles'] = [200, 220, 0, 100]
    r = vt.agregar_features_candidatas(df)
    assert r.iloc[1].crecimiento_demanda_1m == pytest.approx(.2)
    assert r.iloc[1].crecimiento_capacidad_1m == pytest.approx(.1)
    assert r.iloc[1].brecha_crecimiento_demanda_capacidad == pytest.approx(.1)
    assert r.iloc[2].crecimiento_demanda_1m == -1
    assert r.iloc[3][vt.CRECIMIENTO].isna().all()
    assert not np.isinf(r[vt.CRECIMIENTO].to_numpy()).any()


def test_catalogo_sin_duplicados_y_fuentes_definidas():
    definiciones = vt.definiciones_features()
    assert set(definiciones) == set(vt.NUEVAS_FEATURES)
    assert len(vt.NUEVAS_FEATURES) == len(set(vt.NUEVAS_FEATURES))
    assert set(vt.REUTILIZADAS).isdisjoint(vt.NUEVAS_FEATURES)
    assert 'max_presion_ingresos_3m' not in vt.NUEVAS_FEATURES
    for nombre, definicion in definiciones.items():
        assert definicion['formula'] and definicion['variables_necesarias']
        assert definicion['conjuntos']
    # La base conserva sus valores, incluso su semántica original de faltantes.
    df = historia().assign(tendencia_ocupacion_1m=123, promedio_movil_3m_ocupacion=456)
    r = vt.agregar_features_candidatas(df)
    pd.testing.assert_frame_equal(df, r[df.columns])


def test_prioridad_global_sobre_recall_y_reporte_tradeoff():
    r = resultados()
    r.loc[r.modelo.eq('A'), ['f1_macro', 'balanced_accuracy', 'recall_alto', 'tasa_falsos_negativos_alto']] = [.699, .711, .73, .27]
    tabla, elegido = ef.seleccionar_conjunto(r)
    assert elegido == 'A'  # Mejora global antes que C (solo Recall).
    assert tabla.set_index('modelo').loc['C', 'recall_mejora_con_caida_global']


def test_permutacion_solo_test_no_refit_y_no_causal():
    df = datos('2023-01', '2023-12')
    motor = motor_falso()
    X = df[motor.COLUMNAS_PREDICTORAS]
    modelo = motor.crear_pipeline(X, None)
    original = X.copy(deep=True)
    imp = ef.importancia_permutacion(modelo, X, df[ef.bt.OBJETIVO], 2023, max_muestras=30)
    assert imp.n_muestra.eq(30).all()
    assert not any(e == 'fit' for e, _, _ in motor.llamadas)
    assert imp.set_index('variable').loc['ocupacion_estimada', 'importancia_f1_macro_media'] > 0
    assert imp.set_index('variable').loc['distrito', 'importancia_f1_macro_media'] == 0
    assert imp.es_causal.eq(False).all()
    pd.testing.assert_frame_equal(X, original)
    with pytest.raises(ValueError):
        ef.importancia_permutacion(modelo, X, df[ef.bt.OBJETIVO], 2025)


def test_ablacion_geografia_condicional_y_columnas_individuales():
    cols = ['ratio_camas_disponibles', 'presion_ingresos_camas', 'anio', 'departamento', 'distrito']
    imp = pd.DataFrame({'variable': ['departamento', 'distrito'], 'importancia_f1_macro_media': [0, .02]})
    plan = ef.plan_ablacion(cols, imp)
    assert len(plan) == 4
    assert plan['sin_geograficas_bajo_aporte'] == ['departamento']
    assert plan['sin_ratio_camas_disponibles'] == ['ratio_camas_disponibles']
    assert plan['sin_presion_ingresos_camas'] == ['presion_ingresos_camas']
    assert plan['sin_anio'] == ['anio']


def test_ganador_no_base_congelado_antes_de_2025(tmp_path, monkeypatch):
    original = ef._ajustar_evaluar
    eventos2025 = []
    def observar(motor, algoritmo, df, train, test, columnas, anio):
        if anio == 2025:
            congelado = json.loads((tmp_path/ef.ARCHIVOS[2]).read_text())
            assert congelado['conjunto_seleccionado'] == 'A'
            assert congelado['2025_participo_en_seleccion'] is False
            eventos2025.append(columnas.copy())
        modelo, metricas = original(motor, algoritmo, df, train, test, columnas, anio)
        # Solo resultado sintético histórico: fuerza A como ganador sin entrenar XGB.
        if anio < 2025 and 'ocupacion_lag_1m' not in columnas:
            metricas.update(f1_macro=.5, balanced_accuracy=.5, recall_alto=.5, tasa_falsos_negativos_alto=.5)
        return modelo, metricas
    monkeypatch.setattr(ef, '_ajustar_evaluar', observar)
    _, _, reporte = ef.evaluar_features(datos(), tmp_path, motor=motor_falso())
    assert reporte['conjunto_seleccionado'] == 'A'
    assert len(eventos2025) == 2
    assert 'ocupacion_lag_1m' not in eventos2025[0]
    assert 'ocupacion_lag_1m' in eventos2025[1]
    assert reporte['ablacion_cambia_seleccion'] is False
