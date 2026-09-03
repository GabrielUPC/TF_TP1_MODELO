from types import SimpleNamespace
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtesting_temporal as bt
from src.variables_temporales import COLUMNAS_TEMPORALES, agregar_variables_temporales


def datos():
    filas = []
    for periodo in pd.period_range('2018-01', '2022-12', freq='M'):
        for clase in (0, 1, 2):
            filas.append({'codigo_ipress': str(clase), 'servicio_hospitalizacion': 'S',
                          'periodo_actual': str(periodo-1), 'periodo_predicho': str(periodo),
                          'ocupacion_estimada': [0.5, 0.75, 0.9][clase],
                          'nivel_riesgo_actual_codificado': clase, bt.OBJETIVO: clase})
    df = pd.DataFrame(filas)
    df.index = df.index*7+11
    return df


def motor_prueba():
    eventos = []
    class Modelo:
        classes_ = np.array([2, 0, 1])  # Obliga a ordenar probabilidades.
        def __init__(self, nombre):
            self.nombre = nombre
        def predict(self, X):
            eventos.append(('predict', self.nombre, X.index.tolist()))
            if self.nombre == 'XGBoost':
                return np.zeros(len(X), dtype=int)
            return np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
        def predict_proba(self, X):
            pred = np.select([X.ocupacion_estimada.ge(.85), X.ocupacion_estimada.ge(.70)], [2, 1], default=0)
            return np.eye(3)[pred][:, self.classes_]
    def crear(X, algoritmo):
        eventos.append(('crear', algoritmo, X.index.tolist()))
        return Modelo(algoritmo)
    def ajustar(nombre, modelo, X, y):
        assert X.index.equals(y.index)
        eventos.append(('fit', nombre, X.index.tolist()))
        return modelo
    return SimpleNamespace(COLUMNAS_PREDICTORAS=['ocupacion_estimada'], COLUMNAS_EXCLUIDAS=list(bt.ETIQUETAS),
        obtener_modelos=lambda: {n: n for n in ['Regresion_Logistica', 'Random_Forest', 'XGBoost']},
        crear_pipeline=crear, ajustar_pipeline=ajustar, eventos=eventos)


def test_folds_expansivos_sin_futuro_mismo_anio_y_historial_suficiente():
    df = datos()
    folds, plan = bt.crear_folds_expansivos(df)
    assert [a for a, _, _ in folds] == [2020, 2021, 2022]
    anterior = set()
    for anio, train, test in folds:
        assert all(pd.Period(p, freq='M').year < anio for p in df.loc[train, 'periodo_predicho'])
        assert all(pd.Period(p, freq='M').year == anio for p in df.loc[test, 'periodo_predicho'])
        assert set(train).isdisjoint(test)
        assert anterior < set(train)
        anterior = set(train)
    assert not plan.loc[plan.anio_prueba == 2019, 'elegible'].item()


def test_descarta_fold_sin_tres_clases_controladamente():
    df = datos()
    df.loc[df.periodo_predicho.lt('2020-01'), bt.OBJETIVO] = 0
    folds, plan = bt.crear_folds_expansivos(df)
    assert [a for a, _, _ in folds] == [2021, 2022]
    assert 'cada clase' in plan.loc[plan.anio_prueba == 2020, 'motivo'].item()


def test_no_admite_anio_incompleto_ni_huecos_en_historial_minimo():
    df = datos().query("periodo_predicho != '2021-05'")
    folds, plan = bt.crear_folds_expansivos(df)
    assert [a for a, _, _ in folds] == [2020]
    assert 'incompleto' in plan.loc[plan.anio_prueba == 2021, 'motivo'].item()
    assert 'meses previos' in plan.loc[plan.anio_prueba == 2022, 'motivo'].item()


@pytest.mark.parametrize('caso', ['t_mas_2', 'indice', 'duplicado', 'clase'])
def test_valida_datos_antes_de_ajustar(caso):
    df = datos()
    if caso == 't_mas_2':
        df.loc[df.index[0], 'periodo_predicho'] = '2018-02'
    elif caso == 'indice':
        df.index = [0]*len(df)
    elif caso == 'duplicado':
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    else:
        df.loc[df.index[0], bt.OBJETIVO] = 3
    with pytest.raises(ValueError):
        bt.crear_folds_expansivos(df)


def test_metricas_alto_y_errores_severos_con_denominadores_explicitos():
    m = bt.calcular_metricas_backtesting([2, 2, 2, 2, 1, 0], [2, 1, 0, 2, 2, 0])
    assert m['casos_alto_reales'] == 4
    assert m['recall_alto'] == .5
    assert m['precision_alto'] == pytest.approx(2/3)
    assert m['f1_alto'] == pytest.approx(4/7)
    assert m['falsos_negativos_alto'] == 2
    assert m['tasa_falsos_negativos_alto'] == .5
    assert m['errores_alto_bajo'] == 1
    assert m['proporcion_alto_bajo'] == .25
    assert m['proporcion_alto_bajo_total'] == pytest.approx(1/6)
    assert m['f1_macro'] == pytest.approx(26/63)
    assert m['balanced_accuracy'] == .5
    assert m['specificity_macro'] == pytest.approx((.8+.8+.5)/3)
    assert np.isnan(m['roc_auc_ovr_macro'])


def test_sin_alto_tasas_no_inventan_ceros():
    m = bt.calcular_metricas_backtesting([0, 1], [0, 1])
    assert m['casos_alto_reales'] == 0
    assert np.isnan(m['recall_alto']) and np.isnan(m['tasa_falsos_negativos_alto'])
    assert np.isnan(m['proporcion_alto_bajo'])


@pytest.mark.parametrize('perfecto,esperado', [(True, 1.0), (False, .5)])
def test_auc_probabilidades_y_empates(perfecto, esperado):
    real = [0, 1, 2, 0, 1, 2]
    proba = np.eye(3)[real] if perfecto else np.full((6, 3), 1/3)
    assert bt.calcular_metricas_backtesting(real, real, proba)['roc_auc_ovr_macro'] == esperado


def test_todos_modelos_y_baselines_mismo_test_y_reportes(tmp_path):
    df, motor = datos(), motor_prueba()
    original = df.copy(deep=True)
    sentinel = tmp_path/'modelo_ipress.joblib'
    sentinel.write_bytes(b'produccion intacta')
    metricas, resumen, comparacion = bt.ejecutar_backtesting(df, tmp_path, motor=motor)
    assert len(metricas) == 3*6
    assert len(resumen) == 6
    for anio, train, test in bt.crear_folds_expansivos(df)[0]:
        grupo = metricas.query('anio_prueba == @anio')
        assert grupo.test_sha256.nunique() == 1 and grupo.n_test.nunique() == 1
        for nombre in motor.obtener_modelos():
            assert ('fit', nombre, train.tolist()) in motor.eventos
            assert ('crear', nombre, train.tolist()) in motor.eventos
            assert ('predict', nombre, test.tolist()) in motor.eventos
        esperado = bt.calcular_metricas_backtesting(df.loc[test, bt.OBJETIVO],
                    bt.predicciones_baselines(df.loc[test], df.loc[train, bt.OBJETIVO])['Clase_Mayoritaria'])
        assert grupo.query("modelo == 'Clase_Mayoritaria'").f1_macro.item() == esperado['f1_macro']
    assert metricas.query("modelo == 'Regresion_Logistica'").roc_auc_ovr_macro.eq(1).all()
    assert 'XGBoost' not in comparacion['mejores_modelos']
    assert set(comparacion['mejores_modelos']) == {'Regresion_Logistica', 'Random_Forest'}
    assert comparacion['modelo_produccion_modificado'] is False
    assert set(bt.METRICAS).issubset(pd.read_csv(tmp_path/'metricas_backtesting_temporal.csv').columns)
    assert {'f1_macro_promedio', 'f1_macro_std', 'f1_macro_min', 'f1_macro_max',
            'recall_alto_promedio', 'tasa_falsos_negativos_alto_promedio'}.issubset(
                pd.read_csv(tmp_path/'resumen_backtesting_temporal.csv').columns)
    assert json.loads((tmp_path/'comparacion_backtesting_temporal.json').read_text())['anios_elegibles'] == [2020, 2021, 2022]
    assert sentinel.read_bytes() == b'produccion intacta'
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize('columna', sorted(bt.ETIQUETAS))
def test_rechaza_etiquetas_en_features(columna):
    motor = motor_prueba()
    motor.COLUMNAS_PREDICTORAS.append(columna)
    with pytest.raises(ValueError, match='Columnas objetivo'):
        bt.ejecutar_backtesting(datos(), motor=motor)
    assert not motor.eventos


def test_predictores_reales_no_incluyen_etiquetas():
    # Inspección de las constantes reales sin importar dependencias binarias.
    arbol = ast.parse((bt.ROOT/'src/entrenar_modelo.py').read_text(encoding='utf-8'))
    nombres = {'COLUMNAS_PREDICTORAS_BASE', 'COLUMNAS_PREDICTORAS', 'COLUMNAS_EXCLUIDAS'}
    nodos = [n for n in arbol.body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id in nombres for t in n.targets)]
    ns = {'COLUMNAS_TEMPORALES': COLUMNAS_TEMPORALES}
    exec(compile(ast.Module(body=nodos, type_ignores=[]), '<columnas>', 'exec'), ns)
    assert set(ns['COLUMNAS_PREDICTORAS']).isdisjoint(bt.ETIQUETAS)
    assert set(ns['COLUMNAS_PREDICTORAS']).isdisjoint(ns['COLUMNAS_EXCLUIDAS'])


def test_variables_temporales_invariantes_al_cambiar_futuro():
    df = pd.DataFrame({'codigo_ipress': ['A']*4, 'servicio_hospitalizacion': ['S']*4,
                       'anio': [2024]*4, 'mes': [1, 2, 3, 4], 'ocupacion_estimada': [.5]*4,
                       'presion_ingresos_camas': [2]*4, 'total_ingresos': [20]*4,
                       'total_egresos': [18]*4, 'total_estancias': [60]*4})
    antes = agregar_variables_temporales(df)
    df.loc[3, ['ocupacion_estimada', 'presion_ingresos_camas', 'total_ingresos',
               'total_egresos', 'total_estancias']] = 999
    despues = agregar_variables_temporales(df)
    pd.testing.assert_frame_equal(antes.iloc[:3], despues.iloc[:3])


def test_resumen_no_compara_tests_diferentes():
    metricas, _, _ = bt.ejecutar_backtesting(datos(), motor=motor_prueba())
    metricas.loc[0, 'test_sha256'] = 'otro test'
    with pytest.raises(ValueError, match='mismo test'):
        bt.resumir_backtesting(metricas)

