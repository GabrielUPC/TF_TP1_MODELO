"""Diez reglas sobre las mismas probabilidades de XGBoost D; sin producción."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtesting_temporal as bt
from . import evaluar_features_temporales as ef
from . import optimizar_xgboost as opt
from .variables_temporales_experimentales import CONJUNTOS, agregar_features_candidatas

ANIOS_DESARROLLO = (2018, 2021, 2022, 2023, 2024)
BASE = 'argmax'
MAX_CAIDA_F1 = .02
ARCHIVOS = ('resultados_reglas_decision.csv', 'resumen_reglas_decision.csv',
            'seleccion_regla_decision.json')
ORDEN = ['proporcion_alto_bajo_promedio', 'tasa_falsos_negativos_alto_promedio',
         'recall_alto_promedio', 'f1_macro_promedio', 'balanced_accuracy_promedio']
CRITERIO = {'max_caida_f1_vs_D_argmax': MAX_CAIDA_F1, 'orden': ORDEN,
            'sentido': ['min', 'min', 'max', 'max', 'max'],
            'denominador_proporcion_alto_bajo': 'casos Alto reales del fold',
            'agregacion': 'media aritmética de los cinco años, con igual peso por año',
            'empate': 'preferir argmax; después orden fijo de reglas declaradas',
            'tolerancia_limite_f1': 1e-12}


def reglas_candidatas():
    return [{'regla': BASE, 'tipo_regla': 'argmax', 'umbral': None},
            *[{'regla': f'alto_{u:.2f}', 'tipo_regla': 'alto', 'umbral': u}
              for u in (.25, .30, .35, .40, .45)],
            *[{'regla': f'proteccion_{u:.2f}', 'tipo_regla': 'proteccion', 'umbral': u}
              for u in (.20, .25, .30, .35)]]


def validar_probabilidades(probabilidades):
    p = np.asarray(probabilidades, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-7)):
        raise ValueError('Probabilidades inválidas: se requiere matriz N x 3, orden Bajo/Medio/Alto y suma 1.')
    return p


def aplicar_regla(probabilidades, regla):
    """No muta ni recalibra p. Empates argmax: primera clase en orden 0/1/2."""
    if regla not in reglas_candidatas():
        raise ValueError('Regla fuera del espacio fijo de diez alternativas.')
    p = validar_probabilidades(probabilidades)
    pred = p.argmax(axis=1)
    if regla['tipo_regla'] == 'alto':
        pred[p[:, 2] >= regla['umbral']] = 2
    elif regla['tipo_regla'] == 'proteccion':
        pred[(pred == 0) & (p[:, 2] >= regla['umbral'])] = 1
    return pred


def probabilidades_fold(motor, algoritmo, df, train, test, columnas, anio):
    """Un solo fit y un solo predict_proba para TODAS las reglas del año."""
    if len(columnas) != len(set(columnas)) or set(columnas) & (bt.ETIQUETAS | set(motor.COLUMNAS_EXCLUIDAS)):
        raise ValueError('Predictores duplicados o etiquetas entre features.')
    periodos = pd.Series(pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M').year, index=df.index)
    if not periodos.loc[train].lt(anio).all() or not periodos.loc[test].eq(anio).all():
        raise ValueError('Train debe ser anterior al test y test exclusivo del año indicado.')
    X_train = df.loc[train, columnas].copy()
    y_train = df.loc[train, bt.OBJETIVO]
    pipeline = motor.crear_pipeline(X_train, algoritmo)
    pipeline.fit(X_train, y_train, modelo__sample_weight=opt.sample_weights(y_train, 'balanceado'))
    clases = list(pipeline.classes_)
    if len(clases) != 3 or set(clases) != {0, 1, 2}:
        raise ValueError('El clasificador debe contener exactamente las clases 0/1/2.')
    p = np.asarray(pipeline.predict_proba(df.loc[test, columnas].copy()))
    if p.shape != (len(test), 3):
        raise ValueError('Las probabilidades no cubren exactamente el test.')
    p = validar_probabilidades(p[:, [clases.index(c) for c in (0, 1, 2)]]).copy()
    p.setflags(write=False)
    return p


def evaluar_probabilidades(df, train, test, p, anio, reglas, fase):
    p = validar_probabilidades(p)
    if len(p) != len(test):
        raise ValueError('El número de probabilidades y filas de test no coincide.')
    huella = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
    filas = []
    for regla in reglas:
        metricas = bt.calcular_metricas_backtesting(df.loc[test, bt.OBJETIVO], aplicar_regla(p, regla), p)
        fila = opt._registro(df, test, train, regla['regla'], anio, metricas)
        fila['regla'] = fila.pop('configuracion')
        fila.update(tipo='regla_decision', tipo_regla=regla['tipo_regla'], umbral=regla['umbral'],
                    fase=fase, probabilidades_sha256=huella)
        filas.append(fila)
    return pd.DataFrame(filas)


def seleccionar_regla(resultados):
    reglas = reglas_candidatas()
    nombres = [r['regla'] for r in reglas]
    if resultados.empty or set(resultados.regla) != set(nombres):
        raise ValueError('La comparación requiere las diez reglas, incluido argmax.')
    if 'fase' in resultados and not resultados.fase.eq('desarrollo').all():
        raise ValueError('Solo desarrollo puede intervenir en la selección.')
    for _, grupo in resultados.groupby('regla'):
        if set(grupo.anio_prueba) != set(ANIOS_DESARROLLO):
            raise ValueError('Se requieren los cinco años históricos; 2025 no participa en selección.')
    for _, grupo in resultados.groupby('anio_prueba'):
        for c in ['probabilidades_sha256', 'test_sha256', 'n_test']:
            if grupo[c].isna().any() or grupo[c].nunique() != 1:
                raise ValueError('Todas las reglas deben compartir test y probabilidades.')
    resumen = bt.resumir_backtesting(resultados.rename(columns={'regla': 'modelo'})).rename(columns={'modelo': 'regla'})
    if not np.isfinite(resumen[ORDEN].to_numpy()).all():
        raise ValueError('Faltan métricas válidas para seleccionar.')
    referencia = resumen.set_index('regla').loc[BASE]
    for metrica in ORDEN:
        resumen['delta_' + metrica] = resumen[metrica] - referencia[metrica]
    resumen['admisible'] = resumen.f1_macro_promedio.ge(referencia.f1_macro_promedio-MAX_CAIDA_F1-1e-12)
    indices = resumen.loc[resumen.admisible].assign(_orden=lambda d: d.regla.map({r: i for i, r in enumerate(nombres)}))
    indices = indices.sort_values([*ORDEN, '_orden'], ascending=[True, True, False, False, False, True], kind='stable').index
    resumen['ranking'] = pd.Series({i: pos+1 for pos, i in enumerate(indices)}, dtype='Int64')
    elegido = resumen.loc[indices[0], 'regla']
    resumen['seleccionada'] = resumen.regla.eq(elegido)
    return resumen, elegido


def evaluar_reglas(df, output_dir, *, motor=None, procedencia=None):
    destino = Path(output_dir)
    if any((destino / archivo).exists() for archivo in ARCHIVOS):
        raise FileExistsError('Ya existe este experimento; no se sobrescribe ni se repite 2025 automáticamente.')
    historico, folds, plan = ef.preparar_desarrollo(df)
    if tuple(f[0] for f in folds) != ANIOS_DESARROLLO:
        raise ValueError('El plan no coincide con los años históricos fijados.')
    motor = motor if motor is not None else bt._motor_existente()
    columnas = [*motor.COLUMNAS_PREDICTORAS, *CONJUNTOS['D']]
    algoritmo = motor.obtener_modelos().get('XGBoost')
    if algoritmo is None:
        raise RuntimeError('XGBoost no está disponible.')
    reglas = reglas_candidatas()
    filas = []
    for anio, train, test in folds:
        print(f'XGBoost D {anio}: un ajuste, diez decisiones sobre el mismo test', flush=True)
        p = probabilidades_fold(motor, algoritmo, historico, train, test, columnas, anio)
        filas.append(evaluar_probabilidades(historico, train, test, p, anio, reglas, 'desarrollo'))
    resultados = pd.concat(filas, ignore_index=True)
    resumen, nombre = seleccionar_regla(resultados)
    elegida = next(r for r in reglas if r['regla'] == nombre)
    tabla = resumen.set_index('regla')
    reporte = {'version_experimento': 'reglas_decision_D_v1',
        'estado': 'seleccion_historica_congelada_antes_de_2025',
        'reglas_probadas': reglas, 'anios_desarrollo': list(ANIOS_DESARROLLO),
        'regla_seleccionada': elegida, 'criterio_seleccion': CRITERIO,
        'metricas_promedio': tabla.loc[nombre].to_dict(), 'comparacion_argmax': tabla.loc[BASE].to_dict(),
        'metricas_por_regla': resumen.to_dict(orient='records'),
        'conjunto_features': 'D', 'columnas_predictoras': columnas,
        'hiperparametros_sin_modificar': algoritmo.get_params(),
        'pesos': 'balanceado, calculado solo en train; mismo pipeline para todas las reglas por año',
        'plan_folds': plan.to_dict(orient='records'), 'procedencia': procedencia or {},
        '2025_participo_en_seleccion': False, 'evaluacion_2025': None,
        'advertencia_2025': 'Ya observado previamente: comprobación adicional, no holdout virgen.',
        'limitaciones': ['Las probabilidades no se recalibran: los umbrales son decisiones experimentales.',
            'Protección Bajo->Medio reduce errores severos, pero por sí sola no mejora Recall Alto ni FNR Alto.',
            'La promoción a Alto puede reducir precisión: se reportan precision_alto y f1_alto.',
            'Años históricos reutilizados en experimentos previos: posible sobreajuste por selección.',
            'No se aplica la regla a la API ni se guardan modelos de producción.'],
        'es_modelo_final_produccion': False}
    destino.mkdir(parents=True, exist_ok=True)
    ruta_seleccion = destino / ARCHIVOS[2]
    opt._json(ruta_seleccion, reporte, exclusivo=True)
    resultados.to_csv(destino / ARCHIVOS[0], index=False)
    resumen.to_csv(destino / ARCHIVOS[1], index=False)

    # Primera consulta a etiquetas de 2025, siempre después de congelar la selección.
    periodos = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    final = df.loc[periodos.year <= 2025].copy()
    folds_final, _ = bt.crear_folds_expansivos(final)
    fold = next((f for f in folds_final if f[0] == 2025), None)
    if fold is None:
        raise ValueError('2025 no es elegible. Selección histórica conservada.')
    final = agregar_features_candidatas(final)
    anio, train, test = fold
    reporte['estado'] = 'comprobacion_2025_iniciada_no_repetir'
    opt._json(ruta_seleccion, reporte)
    print('XGBoost D 2025: un ajuste; comparar argmax y regla congelada', flush=True)
    p = probabilidades_fold(motor, algoritmo, final, train, test, columnas, anio)
    comparadas = [reglas[0]] if nombre == BASE else [reglas[0], elegida]
    externas = evaluar_probabilidades(final, train, test, p, anio, comparadas, 'comprobacion_2025')
    comparacion = externas.set_index('regla')
    reporte.update(estado='completado_sin_produccion', evaluacion_2025={
        'argmax': comparacion.loc[BASE].to_dict(), 'seleccionada': comparacion.loc[nombre].to_dict(),
        'regla_seleccionada': elegida, 'n_ajustes_XGBoost_D': 1})
    resultados = pd.concat([resultados, externas], ignore_index=True)
    resultados.to_csv(destino / ARCHIVOS[0], index=False)
    opt._json(ruta_seleccion, reporte)
    return resultados, resumen, reporte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-plan', action='store_true', help='Verificar años sin ajustar modelos ni escribir resultados')
    args = parser.parse_args()
    ruta = bt.ROOT / 'data/processed/dataset_modelo_ipress.csv'
    contenido = ruta.read_bytes()
    huella = hashlib.sha256(contenido).hexdigest()
    metadata = json.loads((ruta.parent / 'dataset_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset_sha256') != huella:
        raise ValueError('El dataset no coincide con su metadata.')
    df = pd.read_csv(io.BytesIO(contenido), dtype={'codigo_ipress': str})
    if args.solo_plan:
        _, folds, _ = ef.preparar_desarrollo(df)
        print(json.dumps({'reglas': reglas_candidatas(), 'anios_desarrollo': [f[0] for f in folds],
            'comprobacion_adicional': 2025, 'ajustes_totales_previstos': len(folds)+1,
            'conjunto': 'D', 'dataset_sha256': huella}, ensure_ascii=False, indent=2))
    else:
        _, resumen, reporte = evaluar_reglas(df, bt.ROOT / 'models', procedencia={'dataset_sha256': huella})
        print(resumen[['regla', *ORDEN, 'admisible', 'ranking']].to_string(index=False))
        print(json.dumps({'regla_seleccionada': reporte['regla_seleccionada'],
                          'evaluacion_2025': reporte['evaluacion_2025']}, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
