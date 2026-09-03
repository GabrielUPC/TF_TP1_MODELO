"""Búsqueda temporal limitada de XGBoost; sin guardar modelos de producción."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import backtesting_temporal as bt
else:
    import backtesting_temporal as bt

HOLDOUT = 2025
TOL_RECALL = 0.01
TOL_F1_SEGURIDAD = 0.02
PERFILES = {
    'base': dict(n_estimators=300, max_depth=5, learning_rate=.05, min_child_weight=1,
                 subsample=.8, colsample_bytree=.8, gamma=0, reg_alpha=0, reg_lambda=1),
    'regularizado': dict(n_estimators=200, max_depth=3, learning_rate=.08, min_child_weight=3,
                        subsample=.9, colsample_bytree=.9, gamma=.1, reg_alpha=.1, reg_lambda=3),
    'profundo': dict(n_estimators=400, max_depth=6, learning_rate=.03, min_child_weight=5,
                    subsample=.75, colsample_bytree=.75, gamma=.2, reg_alpha=.3, reg_lambda=5),
}
PESOS = {'balanceado': None, 'alto_1.1': {0: 1., 1: 1., 2: 1.1},
         'alto_1.2': {0: 1., 1: 1., 2: 1.2}, 'alto_1.3': {0: 1., 1: 1., 2: 1.3},
         'alto_1.5': {0: 1., 1: 1., 2: 1.5}}
BASE = 'base__balanceado'
PERSISTENCIA = 'Persistencia'
COLUMNAS_RANKING = ['f1_macro_promedio', 'balanced_accuracy_promedio',
                    'recall_alto_promedio', 'tasa_falsos_negativos_alto_promedio']


def configuraciones():
    return [{'configuracion': f'{perfil}__{peso}', 'parametros': {**parametros, 'random_state': 42},
             'esquema_pesos': peso, 'pesos_clase': pesos}
            for perfil, parametros in PERFILES.items() for peso, pesos in PESOS.items()]


def sample_weights(y_train, esquema):
    y = bt._clases(y_train)
    if len(y) == 0 or set(y) != {0, 1, 2}:
        raise ValueError('Los pesos se calculan sobre train con las tres clases.')
    if esquema not in PESOS:
        raise ValueError('Esquema de pesos desconocido.')
    if esquema == 'balanceado':
        conteos = np.bincount(y, minlength=3)
        return len(y)/(3*conteos[y])
    # Pesos absolutos solicitados, no multiplicadores del balanceado.
    return np.array([PESOS[esquema][int(clase)] for clase in y], dtype=float)


def historial_tuning(df, holdout=HOLDOUT):
    periodos = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    historico = df.loc[periodos.year < holdout].copy()
    folds, plan = bt.crear_folds_expansivos(historico)
    if not folds:
        raise ValueError('No existen folds históricos elegibles anteriores al holdout.')
    return historico, folds, plan


def rankings(resultados):
    if resultados.empty or resultados.duplicated(['configuracion', 'anio_prueba']).any():
        raise ValueError('Resultados vacíos o duplicados.')
    if BASE not in set(resultados.configuracion) or PERSISTENCIA not in set(resultados.configuracion):
        raise ValueError('La comparación requiere XGBoost base y Persistencia.')
    if resultados.anio_prueba.ge(HOLDOUT).any():
        raise ValueError('El holdout no puede participar en el ranking.')
    anios = resultados.groupby('configuracion').anio_prueba.agg(lambda x: frozenset(x))
    if len(set(anios)) != 1:
        raise ValueError('Todas las configuraciones deben usar los mismos años.')
    for _, grupo in resultados.groupby('anio_prueba'):
        if grupo.test_sha256.nunique() != 1 or grupo.n_test.nunique() != 1:
            raise ValueError('Los registros de validación deben ser idénticos.')
    resumen = resultados.groupby('configuracion')[bt.METRICAS].agg(['mean', 'std', 'min', 'max'])
    resumen.columns = [f'{m}_{dict(mean="promedio", std="std", min="min", max="max")[s]}'
                       for m, s in resumen.columns]
    resumen = resumen.reset_index()
    resumen['n_anios'] = len(resultados.anio_prueba.unique())
    base = resumen.set_index('configuracion').loc[BASE]
    for metrica in COLUMNAS_RANKING:
        if not np.isfinite(resumen[metrica]).all():
            raise ValueError('Faltan métricas válidas para comparar configuraciones.')
        resumen['delta_'+metrica] = resumen[metrica]-base[metrica]
    candidatos = resumen.configuracion.ne(PERSISTENCIA)
    resumen['admisible_principal'] = candidatos & (
        resumen.recall_alto_promedio >= base.recall_alto_promedio-TOL_RECALL)
    resumen['admisible_seguridad'] = candidatos & (
        resumen.f1_macro_promedio >= base.f1_macro_promedio-TOL_F1_SEGURIDAD)
    def ordenar(mascara, columnas, asc):
        # Preferir base en empate total evita complejidad sin mejora observada.
        tabla = resumen.loc[mascara].assign(_base=lambda d: d.configuracion.ne(BASE))
        return tabla.sort_values([*columnas, '_base', 'configuracion'],
            ascending=[*asc, True, True], kind='stable').index.tolist()
    principal = ordenar(resumen.admisible_principal, COLUMNAS_RANKING, [False, False, False, True])
    seguridad = ordenar(resumen.admisible_seguridad,
        ['recall_alto_promedio', 'tasa_falsos_negativos_alto_promedio',
         'f1_macro_promedio', 'balanced_accuracy_promedio'], [False, True, False, False])
    resumen['ranking_principal'] = pd.Series({i: n+1 for n, i in enumerate(principal)}, dtype='Int64')
    resumen['ranking_seguridad'] = pd.Series({i: n+1 for n, i in enumerate(seguridad)}, dtype='Int64')
    return resumen, resumen.loc[principal[0], 'configuracion'], resumen.loc[seguridad[0], 'configuracion']


def _evaluar(motor, algoritmo, df, train, test, esquema=None):
    columnas = list(motor.COLUMNAS_PREDICTORAS)
    if set(columnas) & (bt.ETIQUETAS | set(motor.COLUMNAS_EXCLUIDAS)):
        raise ValueError('No se admiten columnas objetivo entre predictores.')
    X_train, X_test = df.loc[train, columnas].copy(), df.loc[test, columnas].copy()
    y_train = df.loc[train, bt.OBJETIVO]
    modelo = motor.crear_pipeline(X_train, algoritmo)
    if esquema is None:
        modelo.fit(X_train, y_train)
    else:
        modelo.fit(X_train, y_train, modelo__sample_weight=sample_weights(y_train, esquema))
    pred = modelo.predict(X_test)
    proba = np.asarray(modelo.predict_proba(X_test))
    clases = list(modelo.classes_)
    return bt.calcular_metricas_backtesting(df.loc[test, bt.OBJETIVO], pred,
                                           proba[:, [clases.index(c) for c in (0, 1, 2)]])


def _registro(df, test, train, nombre, anio, metricas):
    huella = hashlib.sha256(pd.util.hash_pandas_object(
        df.loc[test, ['codigo_ipress', 'servicio_hospitalizacion', 'periodo_predicho', bt.OBJETIVO]],
        index=True).to_numpy().tobytes()).hexdigest()
    return {'configuracion': nombre, 'anio_prueba': anio, 'n_train': len(train),
            'n_test': len(test), 'test_sha256': huella, **metricas}


def _json(path, datos, *, exclusivo=False):
    # Conversión de tipos numpy y valores no definidos a JSON estándar.
    texto = pd.Series({'resultado': datos}).to_json(force_ascii=False)
    limpio = json.loads(texto)['resultado']
    with Path(path).open('x' if exclusivo else 'w', encoding='utf-8') as archivo:
        archivo.write(json.dumps(limpio, ensure_ascii=False, indent=2, allow_nan=False))


def optimizar(df, output_dir, *, motor=None):
    destino = Path(output_dir)
    bloqueo = destino/'holdout_xgboost_2025.json'
    elegido_path = destino/'mejor_configuracion_xgboost.json'
    if bloqueo.exists() or elegido_path.exists():
        raise FileExistsError('Existe una selección o evaluación previa: no se repite el holdout automáticamente.')
    historico, folds, plan = historial_tuning(df)
    motor = motor if motor is not None else bt._motor_existente()
    algoritmos = motor.obtener_modelos()
    if 'XGBoost' not in algoritmos or 'Random_Forest' not in algoritmos:
        raise RuntimeError('Se requieren XGBoost y Random Forest disponibles.')
    configs = configuraciones()
    filas = []
    for anio, train, test in folds:
        for config in configs:
            print(f'Tuning {anio}: {config["configuracion"]}', flush=True)
            algoritmo = algoritmos['XGBoost'].__class__(**algoritmos['XGBoost'].get_params())
            algoritmo.set_params(**config['parametros'])
            metricas = _evaluar(motor, algoritmo, historico, train, test, config['esquema_pesos'])
            filas.append(_registro(historico, test, train, config['configuracion'], anio, metricas))
        m = bt.calcular_metricas_backtesting(historico.loc[test, bt.OBJETIVO],
                     historico.loc[test, 'nivel_riesgo_actual_codificado'])
        filas.append(_registro(historico, test, train, PERSISTENCIA, anio, m))
    resultados = pd.DataFrame(filas)
    resumen, ganador, seguro = rankings(resultados)
    seleccion = next(c for c in configs if c['configuracion'] == ganador)
    tabla = resumen.set_index('configuracion')
    reporte = {'estado': 'seleccion_congelada_antes_de_holdout',
        'configuracion_elegida': seleccion,
        'alternativa_seguridad_no_evaluada_en_holdout': next(c for c in configs if c['configuracion'] == seguro),
        'anios_tuning': [f[0] for f in folds], 'anio_holdout': HOLDOUT,
        'espacio_configuraciones': configs, 'criterio_seleccion': {
            'principal': COLUMNAS_RANKING, 'sentido': ['max', 'max', 'max', 'min'],
            'max_caida_recall_vs_base': TOL_RECALL, 'max_caida_f1_seguridad_vs_base': TOL_F1_SEGURIDAD,
            'seleccion_holdout': 'Solo ganador principal; no se elige usando 2025'},
        'metricas_promedio_tuning': tabla.loc[ganador].to_dict(),
        'comparacion_base_tuning': tabla.loc[BASE].to_dict(),
        'comparacion_persistencia_tuning': tabla.loc[PERSISTENCIA].to_dict(),
        'es_modelo_final_produccion': False,
        'advertencia_holdout': '2025 ya fue observado en backtesting previo; no se usa para seleccionar esta búsqueda.'}
    destino.mkdir(parents=True, exist_ok=True)
    _json(elegido_path, reporte, exclusivo=True)  # Congela antes del holdout, también ante concurrencia.
    resultados.to_csv(destino/'resultados_tuning_xgboost.csv', index=False)
    resumen.to_csv(destino/'resumen_tuning_xgboost.csv', index=False)
    periodos = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    final = df.loc[periodos.year <= HOLDOUT].copy()
    folds_final, _ = bt.crear_folds_expansivos(final)
    fold = next((f for f in folds_final if f[0] == HOLDOUT), None)
    if fold is None:
        raise ValueError('2025 no es un año holdout elegible. Selección histórica conservada.')
    anio, train, test = fold
    # Creación exclusiva: ni reruns ni procesos concurrentes repiten el holdout.
    with bloqueo.open('x', encoding='utf-8') as archivo:
        json.dump({'estado': 'iniciado', 'configuracion': ganador, 'anio': anio}, archivo)
    evaluaciones = []
    for nombre, config in [('XGBoost_seleccionado', seleccion), ('XGBoost_base', configs[0])]:
        if nombre == 'XGBoost_base' and ganador == BASE:
            metricas = {k: evaluaciones[0][k] for k in bt.METRICAS}
        else:
            algoritmo = algoritmos['XGBoost'].__class__(**algoritmos['XGBoost'].get_params())
            algoritmo.set_params(**config['parametros'])
            metricas = _evaluar(motor, algoritmo, final, train, test, config['esquema_pesos'])
        evaluaciones.append(_registro(final, test, train, nombre, anio, metricas))
    metricas = _evaluar(motor, algoritmos['Random_Forest'], final, train, test)
    evaluaciones.append(_registro(final, test, train, 'Random_Forest', anio, metricas))
    metricas = bt.calcular_metricas_backtesting(final.loc[test, bt.OBJETIVO],
                                               final.loc[test, 'nivel_riesgo_actual_codificado'])
    evaluaciones.append(_registro(final, test, train, PERSISTENCIA, anio, metricas))
    pd.DataFrame(evaluaciones).to_csv(destino/'evaluacion_holdout_xgboost_2025.csv', index=False)
    reporte.update(estado='evaluacion_holdout_completada', resultados_holdout=evaluaciones)
    _json(elegido_path, reporte)
    _json(bloqueo, {'estado': 'completado', 'configuracion': ganador, 'anio': anio})
    return resultados, resumen, reporte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-plan', action='store_true')
    args = parser.parse_args()
    ruta = bt.ROOT/'data/processed/dataset_modelo_ipress.csv'
    contenido = ruta.read_bytes()
    huella = hashlib.sha256(contenido).hexdigest()
    metadata = json.loads((ruta.parent/'dataset_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset_sha256') != huella:
        raise ValueError('Dataset y metadata no coinciden.')
    df = pd.read_csv(io.BytesIO(contenido), dtype={'codigo_ipress': str})
    historico, folds, plan = historial_tuning(df)
    print(plan.to_string(index=False))
    if args.solo_plan:
        salida = {'anios_tuning': [f[0] for f in folds], 'anio_holdout': HOLDOUT,
                  'dataset_sha256': huella, 'configuraciones': configuraciones(),
                  'estado': 'plan_sin_ajustar_modelos', 'es_modelo_final_produccion': False}
        _json(bt.ROOT/'models/plan_tuning_xgboost.json', salida)
    else:
        _, _, reporte = optimizar(df, bt.ROOT/'models')
        reporte.update(dataset_sha256=huella, definicion_target=metadata.get('definicion_target', {}))
        _json(bt.ROOT/'models/mejor_configuracion_xgboost.json', reporte)
        print(json.dumps(reporte, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
