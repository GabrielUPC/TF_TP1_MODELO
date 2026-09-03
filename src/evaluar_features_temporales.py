"""Experimento Base/A/B/C/D, ablación y comprobación 2025; sin guardar producción."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtesting_temporal as bt
from . import optimizar_xgboost as opt
from .variables_temporales_experimentales import (
    CONJUNTOS, NUEVAS_FEATURES, REUTILIZADAS, ALIAS_EXPERIMENTALES,
    agregar_features_candidatas, definiciones_features,
)

ANIOS_DESARROLLO = (2018, 2021, 2022, 2023, 2024)
MAX_CAIDA_F1 = .02
ARCHIVOS = ('resultados_features_temporales.csv', 'resumen_features_temporales.csv',
            'seleccion_features_temporales.json', 'importancia_features_temporales.csv')
GEOGRAFICAS = ('departamento', 'provincia', 'distrito')
CRITERIO = {
    'prioridad_1': 'Mejorar F1 macro promedio o balanced accuracy promedio',
    'prioridad_2': 'Solo si no hay candidato de prioridad 1: mejorar Recall Alto o reducir FNR Alto',
    'max_caida_f1_absoluta': MAX_CAIDA_F1,
    'orden_dentro_de_prioridad': bt.ORDEN_COMPARACION,
    'sentido': ['max', 'max', 'max', 'min'], 'tolerancia_numerica': 1e-12,
    'empate': 'A, B, C, D; sin mejora admisible conservar Base',
    'ablacion': 'Diagnóstico posterior a selección; no cambia el ganador ni producción',
}
RIESGOS_METODOLOGICOS = [
    '2025 ya observado: comprobación adicional, no holdout virgen; excluido de selección e importancia.',
    'El CSV procesado ya omite meses sin pareja t+1. No se recuperan desde RAW; historia limitada.',
    'Las variables actuales reutilizadas conservan promedios parciales y tendencias cero ante huecos.',
    'Márgenes son transformaciones deterministas de ocupación: colineales, no añaden información independiente.',
    'Conteos de riesgo usan etiqueta ACTUAL observada, nunca futura; solo están permitidos en este experimento.',
    'Importancia por permutación no es causal; predictores correlacionados pueden repartirse o enmascarar aporte.',
    'Ablación retira columnas individuales; derivados correlacionados permanecen y pueden sustituir su aporte.',
    'Los años de desarrollo ya se han usado para otros experimentos: posible sobreajuste de selección.',
    'Se supone disponibilidad de todos los datos de t al cierre; latencias o revisiones retrospectivas no verificadas.',
]


def preparar_desarrollo(df):
    periodos = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    historial = df.loc[periodos.year < 2025].copy()
    folds, plan = bt.crear_folds_expansivos(historial)
    folds = [fold for fold in folds if fold[0] in ANIOS_DESARROLLO]
    if tuple(f[0] for f in folds) != ANIOS_DESARROLLO:
        raise ValueError('No son elegibles todos los años de desarrollo fijados; revise el plan.')
    return agregar_features_candidatas(historial), folds, plan


def seleccionar_conjunto(resultados):
    if set(resultados.modelo) != set(CONJUNTOS):
        raise ValueError('Se requiere comparar exactamente Base, A, B, C y D.')
    for _, grupo in resultados.groupby('modelo'):
        if set(grupo.anio_prueba) != set(ANIOS_DESARROLLO):
            raise ValueError('Solo se permiten los cinco años históricos fijados; nunca 2025.')
    resumen = bt.resumir_backtesting(resultados)
    if not np.isfinite(resumen[bt.ORDEN_COMPARACION].to_numpy()).all():
        raise ValueError('Faltan métricas válidas para seleccionar.')
    base = resumen.set_index('modelo').loc['Base']
    for metrica in bt.ORDEN_COMPARACION:
        resumen['delta_' + metrica] = resumen[metrica] - base[metrica]
    global_mejora = ((resumen.delta_f1_macro_promedio > 1e-12)
                    | (resumen.delta_balanced_accuracy_promedio > 1e-12))
    alto_mejora = ((resumen.delta_recall_alto_promedio > 1e-12)
                  | (resumen.delta_tasa_falsos_negativos_alto_promedio < -1e-12))
    resumen['prioridad'] = np.select([global_mejora, alto_mejora], [1, 2], default=3)
    resumen['admisible'] = ((global_mejora | alto_mejora)
        & (resumen.delta_f1_macro_promedio >= -MAX_CAIDA_F1-1e-12))
    resumen.loc[resumen.modelo.eq('Base'), 'admisible'] = True
    resumen['recall_mejora_con_caida_global'] = (resumen.delta_recall_alto_promedio.gt(1e-12)
        & (resumen.delta_f1_macro_promedio.lt(-1e-12) | resumen.delta_balanced_accuracy_promedio.lt(-1e-12)))
    candidatos = resumen.loc[resumen.admisible & resumen.modelo.ne('Base')]
    elegido = 'Base' if candidatos.empty else candidatos.sort_values(
        ['prioridad', *bt.ORDEN_COMPARACION, 'modelo'],
        ascending=[True, False, False, False, True, True], kind='stable').iloc[0].modelo
    resumen['seleccionado'] = resumen.modelo.eq(elegido)
    return resumen, elegido


def _ajustar_evaluar(motor, algoritmo, df, train, test, columnas, anio):
    if set(columnas) & (bt.ETIQUETAS | set(motor.COLUMNAS_EXCLUIDAS)):
        raise ValueError('No se admiten etiquetas entre predictores.')
    periodos = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    anios = pd.Series(periodos.year, index=df.index)
    if not anios.loc[train].lt(anio).all() or not anios.loc[test].eq(anio).all():
        raise ValueError('Train debe ser anterior y test exclusivo del año indicado.')
    X_train, X_test = df.loc[train, columnas].copy(), df.loc[test, columnas].copy()
    y_train = df.loc[train, bt.OBJETIVO]
    modelo = motor.crear_pipeline(X_train, algoritmo)
    modelo.fit(X_train, y_train, modelo__sample_weight=opt.sample_weights(y_train, 'balanceado'))
    pred = modelo.predict(X_test)
    clases = list(modelo.classes_)
    proba = np.asarray(modelo.predict_proba(X_test))[:, [clases.index(c) for c in (0, 1, 2)]]
    return modelo, bt.calcular_metricas_backtesting(df.loc[test, bt.OBJETIVO], pred, proba)


def importancia_permutacion(modelo, X_test, y_test, anio, repeticiones=3, max_muestras=1000):
    """Permutar columnas originales solo en validación histórica; sin reajustar."""
    if anio not in ANIOS_DESARROLLO or repeticiones < 1 or max_muestras < 1:
        raise ValueError('Importancia solo en desarrollo histórico con límites positivos.')
    if not X_test.index.equals(y_test.index):
        raise ValueError('X/y deben compartir registros y orden.')
    X = X_test.sample(n=min(len(X_test), max_muestras), random_state=42).copy()
    y = y_test.loc[X.index]
    original = bt.calcular_metricas_backtesting(y, modelo.predict(X))
    rng = np.random.default_rng(42)
    filas = []
    for columna in X.columns:
        caidas = []
        for _ in range(repeticiones):
            permutado = X.copy()
            permutado[columna] = X[columna].iloc[rng.permutation(len(X))].to_numpy()
            metrica = bt.calcular_metricas_backtesting(y, modelo.predict(permutado))
            caidas.append(original['f1_macro'] - metrica['f1_macro'])
        filas.append({'variable': columna, 'anio_prueba': anio,
            'importancia_f1_macro_media': float(np.mean(caidas)),
            'importancia_f1_macro_std': float(np.std(caidas)),
            'f1_macro_sin_permutar': original['f1_macro'], 'n_test': len(X_test),
            'n_muestra': len(X), 'repeticiones': repeticiones, 'random_state': 42,
            'metodo': 'permutacion_en_test_temporal', 'es_causal': False})
    return pd.DataFrame(filas)


def plan_ablacion(columnas, importancia):
    pruebas = {f'sin_{c}': [c] for c in ('ratio_camas_disponibles', 'presion_ingresos_camas', 'anio') if c in columnas}
    medias = importancia.groupby('variable').importancia_f1_macro_media.mean()
    geos = [c for c in GEOGRAFICAS if c in columnas and c in medias and medias[c] <= 0]
    if geos:
        pruebas['sin_geograficas_bajo_aporte'] = geos
    return pruebas


def _registro(df, train, test, nombre, anio, metricas, columnas, fase):
    fila = opt._registro(df, test, train, nombre, anio, metricas)
    fila['modelo'] = fila.pop('configuracion')
    fila.update(tipo='modelo', fase=fase, n_features=len(columnas))
    return fila


def evaluar_features(df, output_dir, *, motor=None, procedencia=None):
    destino = Path(output_dir)
    if any((destino / nombre).exists() for nombre in ARCHIVOS):
        raise FileExistsError('Ya existe este experimento; no se sobrescriben resultados ni se repite 2025.')
    historico, folds, plan = preparar_desarrollo(df)
    motor = motor if motor is not None else bt._motor_existente()
    base = list(motor.COLUMNAS_PREDICTORAS)
    if (set(base) & (bt.ETIQUETAS | set(motor.COLUMNAS_EXCLUIDAS) | set(NUEVAS_FEATURES))
            or len(base) != len(set(base))):
        raise ValueError('Features base no válidas o candidatas ya presentes en producción.')
    algoritmo = motor.obtener_modelos().get('XGBoost')
    if algoritmo is None:
        raise RuntimeError('XGBoost no está disponible.')
    filas = []
    for anio, train, test in folds:
        for nombre, nuevas in CONJUNTOS.items():
            print(f'Features {anio}: {nombre}', flush=True)
            columnas = [*base, *nuevas]
            _, metricas = _ajustar_evaluar(motor, algoritmo, historico, train, test, columnas, anio)
            filas.append(_registro(historico, train, test, nombre, anio, metricas, columnas, 'desarrollo'))
    resultados = pd.DataFrame(filas)
    resumen, elegido = seleccionar_conjunto(resultados)
    columnas = [*base, *CONJUNTOS[elegido]]
    reporte = {
        'version_experimento': 'features_temporales_abcd_v2',
        'estado': 'seleccion_congelada_antes_de_ablacion_y_2025',
        'features_base': base, 'features_anadidas_por_conjunto': CONJUNTOS,
        'definiciones_variables': definiciones_features(), 'variables_reutilizadas_sin_duplicar': REUTILIZADAS,
        'alias_experimentales': ALIAS_EXPERIMENTALES, 'riesgos_metodologicos': RIESGOS_METODOLOGICOS,
        'posibles_redundantes': ['margen_umbral_alto', 'margen_umbral_medio',
            'distancia_absoluta_umbral_alto', 'historial y cambios correlacionados con ocupacion'],
        'anios_desarrollo': list(ANIOS_DESARROLLO), 'anio_evaluacion_externa': 2025,
        '2025_participo_en_seleccion': False, '2025_evaluado_en_este_experimento': False,
        'metricas_por_conjunto': resumen.to_dict(orient='records'),
        'comparacion_contra_base': resumen.loc[resumen.modelo.eq('Base')].iloc[0].to_dict(),
        'conjunto_seleccionado': elegido, 'criterio_seleccion': CRITERIO,
        'hiperparametros_fijos': algoritmo.get_params(), 'pesos': 'balanceado calculado solo en train',
        'es_modelo_final_produccion': False, 'procedencia': procedencia or {},
        'plan_folds': plan.to_dict(orient='records'),
        'manejo_ausentes': 'NaN sin imputación, ventanas nuevas completas; StandardScaler ajustado solo en train',
    }
    destino.mkdir(parents=True, exist_ok=True)
    seleccion_path = destino / ARCHIVOS[2]
    # Congelar de forma exclusiva. Un fallo posterior exige revisión, nunca un rerun automático de 2025.
    opt._json(seleccion_path, reporte, exclusivo=True)
    resultados.to_csv(destino / ARCHIVOS[0], index=False)
    resumen.to_csv(destino / ARCHIVOS[1], index=False)
    importancias = []
    for anio, train, test in folds:
        print(f'Importancia histórica {anio}: {elegido}', flush=True)
        modelo, _ = _ajustar_evaluar(motor, algoritmo, historico, train, test, columnas, anio)
        importancias.append(importancia_permutacion(modelo, historico.loc[test, columnas],
                                                   historico.loc[test, bt.OBJETIVO], anio))
    importancia = pd.concat(importancias, ignore_index=True)
    importancia['conjunto'] = elegido
    importancia.to_csv(destino / ARCHIVOS[3], index=False)
    ablaciones = plan_ablacion(columnas, importancia)
    registros_ablacion = []
    for nombre, quitar in ablaciones.items():
        reducidas = [c for c in columnas if c not in quitar]
        for anio, train, test in folds:
            print(f'Ablación {anio}: {nombre}', flush=True)
            _, metricas = _ajustar_evaluar(motor, algoritmo, historico, train, test, reducidas, anio)
            fila = _registro(historico, train, test, nombre, anio, metricas, reducidas, 'ablacion')
            referencia = resultados.loc[resultados.modelo.eq(elegido) & resultados.anio_prueba.eq(anio)].iloc[0]
            fila.update({f'delta_{m}_vs_ganador': metricas[m]-referencia[m] for m in bt.METRICAS})
            registros_ablacion.append(fila)
    reporte.update(plan_ablacion=ablaciones, resultados_ablacion=registros_ablacion,
        criterio_geografia='Permutación F1 macro media histórica <= 0; bloque retirado solo como diagnóstico',
        ablacion_cambia_seleccion=False)
    resultados = pd.concat([resultados, pd.DataFrame(registros_ablacion)], ignore_index=True)
    resultados.to_csv(destino / ARCHIVOS[0], index=False)
    opt._json(seleccion_path, reporte)

    # No construir ni consultar las etiquetas de 2025 hasta congelar la selección histórica.
    periodos = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    final = df.loc[periodos.year <= 2025].copy()
    folds_final, _ = bt.crear_folds_expansivos(final)
    fold = next((f for f in folds_final if f[0] == 2025), None)
    if fold is None:
        raise ValueError('2025 no es elegible; selección histórica conservada sin evaluación externa.')
    final = agregar_features_candidatas(final)
    anio, train, test = fold
    reporte['estado'] = 'evaluacion_2025_iniciada_no_repetir'
    opt._json(seleccion_path, reporte)
    externas = []
    for nombre, features in [('Base', base), (f'Seleccionado_{elegido}', columnas)]:
        print(f'Comprobación adicional 2025: {nombre}', flush=True)
        if nombre != 'Base' and elegido == 'Base':
            metricas = {m: externas[0][m] for m in bt.METRICAS}
        else:
            _, metricas = _ajustar_evaluar(motor, algoritmo, final, train, test, features, anio)
        externas.append(_registro(final, train, test, nombre, anio, metricas, features, 'comprobacion_2025'))
    reporte.update(estado='completado_sin_modelo_produccion', evaluacion_2025=externas,
                   **{'2025_evaluado_en_este_experimento': True})
    resultados = pd.concat([resultados, pd.DataFrame(externas)], ignore_index=True)
    resultados.to_csv(destino / ARCHIVOS[0], index=False)
    opt._json(seleccion_path, reporte)
    return resultados, resumen, reporte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-plan', action='store_true', help='No ajusta modelos ni escribe resultados')
    args = parser.parse_args()
    ruta = bt.ROOT / 'data/processed/dataset_modelo_ipress.csv'
    contenido = ruta.read_bytes()
    huella = hashlib.sha256(contenido).hexdigest()
    metadata = json.loads((ruta.parent / 'dataset_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset_sha256') != huella:
        raise ValueError('El dataset no coincide con su metadata.')
    df = pd.read_csv(io.BytesIO(contenido), dtype={'codigo_ipress': str})
    if args.solo_plan:
        historico, folds, plan = preparar_desarrollo(df)
        print(plan.to_string(index=False))
        print(json.dumps({'anios_desarrollo': [f[0] for f in folds], 'conjuntos': CONJUNTOS,
              'ajustes_desarrollo': len(folds)*len(CONJUNTOS), 'ajustes_importancia': len(folds),
              'max_ajustes_ablacion': 4*len(folds), 'max_ajustes_2025': 2,
              'reutilizadas': REUTILIZADAS, 'dataset_sha256': huella,
              'porcentaje_ausentes_candidatas': (historico[NUEVAS_FEATURES].isna().mean()*100).to_dict()},
              ensure_ascii=False, indent=2))
    else:
        _, resumen, reporte = evaluar_features(df, bt.ROOT / 'models', procedencia={
              'dataset_sha256': huella, 'definicion_target': metadata.get('definicion_target', {})})
        print(resumen[['modelo', *bt.ORDEN_COMPARACION, 'admisible', 'seleccionado']].to_string(index=False))
        print('Conjunto seleccionado:', reporte['conjunto_seleccionado'])


if __name__ == '__main__':
    main()
