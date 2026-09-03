"""Cinco variantes de flujo frente a XGBoost D; no guarda modelos de producción."""
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
from .variables_flujo import FEATURES, VARIANTES, agregar_features_flujo, definiciones

ANIOS = (2018, 2021, 2022, 2023, 2024)
ORDEN = [*bt.ORDEN_COMPARACION, 'proporcion_alto_bajo_promedio']
ARCHIVOS = ('resultados_features_flujo.csv', 'resumen_features_flujo.csv', 'seleccion_features_flujo.json')
CRITERIO = {'orden': ORDEN, 'sentido': ['max', 'max', 'max', 'min', 'min'],
    'max_caida_f1_vs_D': .02, 'tolerancia_limite': 1e-12,
    'agregacion': 'Media aritmética de cinco años, igual peso por año',
    'empate': 'Orden fijo D, D+FLUJO, D+DEMANDA, D+PERMANENCIA, D+DINAMICA'}
LIMITACIONES = [
    '2025 ya observado anteriormente: comprobación adicional, no holdout virgen.',
    'Desarrollo reutilizado en otros experimentos: riesgo de sobreajuste de selección.',
    'Historial del D procesado omite meses sin pareja objetivo; no se recuperan desde RAW.',
    'Las features D preexistentes y sus limitaciones se conservan sin cambios.',
    'balance_flujo_mes repite diferencia_ingresos_egresos; crecimiento_pacientes_cama_1m repite crecimiento_demanda_1m en datos válidos.',
    'estancia_promedio_actual comparte fórmula con promedio_estancia, pero devuelve NaN ante egresos cero.',
    'promedio_balance_flujo_3m es balance_flujo_acumulado_3m/3; hay redundancia y colinealidad.',
    'Los balances de flujos no equivalen a censo de pacientes ni miden directamente liberación de camas.',
    'Las variables describen t y pasado; su disponibilidad al predecir t+1 debe estar garantizada.',
]


def preparar_desarrollo(df):
    historico, folds, plan = ef.preparar_desarrollo(df)
    return agregar_features_flujo(historico), folds, plan


def seleccionar(resultados):
    if set(resultados.modelo) != set(VARIANTES) or not resultados.fase.eq('desarrollo').all():
        raise ValueError('Se requieren exactamente las cinco variantes de desarrollo.')
    for _, g in resultados.groupby('modelo'):
        if len(g) != 5 or set(g.anio_prueba) != set(ANIOS):
            raise ValueError('Solo 2018, 2021, 2022, 2023 y 2024; nunca 2025 ni duplicados.')
    for _, g in resultados.groupby('anio_prueba'):
        if g.test_sha256.isna().any() or g.test_sha256.nunique() != 1 or g.n_test.nunique() != 1:
            raise ValueError('Todas las variantes deben compartir test.')
    resumen = bt.resumir_backtesting(resultados)
    if not np.isfinite(resumen[ORDEN].to_numpy()).all():
        raise ValueError('Métricas de selección no evaluables.')
    base = resumen.set_index('modelo').loc['D']
    for m in ORDEN:
        resumen['delta_' + m] = resumen[m]-base[m]
    resumen['admisible'] = resumen.f1_macro_promedio.ge(base.f1_macro_promedio-.02-1e-12)
    candidatos = resumen.loc[resumen.admisible].assign(
        _orden=lambda x: x.modelo.map({v: i for i, v in enumerate(VARIANTES)}))
    indices = candidatos.sort_values([*ORDEN, '_orden'],
        ascending=[False, False, False, True, True, True], kind='stable').index
    elegido = resumen.loc[indices[0], 'modelo']
    resumen['ranking'] = pd.Series({i: n+1 for n, i in enumerate(indices)}, dtype='Int64')
    resumen['seleccionada'] = resumen.modelo.eq(elegido)
    return resumen, elegido


def evaluar(df, destino, *, motor=None, procedencia=None):
    destino = Path(destino)
    if any((destino/n).exists() for n in ARCHIVOS):
        raise FileExistsError('El experimento ya existe; no sobrescribir ni repetir 2025.')
    historico, folds, plan = preparar_desarrollo(df)
    if tuple(f[0] for f in folds) != ANIOS:
        raise ValueError('Años de desarrollo no coinciden con el plan fijo.')
    motor = motor if motor is not None else bt._motor_existente()
    base = [*motor.COLUMNAS_PREDICTORAS, *ef.CONJUNTOS['D']]
    columnas = {n: [*base, *extra] for n, extra in VARIANTES.items()}
    for cols in columnas.values():
        if len(cols) != len(set(cols)) or set(cols) & (bt.ETIQUETAS | set(motor.COLUMNAS_EXCLUIDAS)):
            raise ValueError('Predictores duplicados o etiquetas entre features.')
    algoritmo = motor.obtener_modelos().get('XGBoost')
    if algoritmo is None:
        raise RuntimeError('XGBoost no disponible.')
    def ajustar(datos, fold, nombre, fase):
        anio, train, test = fold
        print(f'Flujo {fase} {anio}: {nombre}', flush=True)
        _, metricas = ef._ajustar_evaluar(motor, algoritmo, datos, train, test, columnas[nombre], anio)
        return ef._registro(datos, train, test, nombre, anio, metricas, columnas[nombre], fase)
    filas = [ajustar(historico, fold, nombre, 'desarrollo') for fold in folds for nombre in VARIANTES]
    resultados = pd.DataFrame(filas)
    resumen, elegido = seleccionar(resultados)
    tabla = resumen.set_index('modelo')
    reporte = {'version': 'features_flujo_v1', 'estado': 'seleccion_congelada_antes_de_2025',
        'variables_evaluadas': FEATURES, 'definiciones_variables': definiciones(),
        'variables_por_variante': columnas, 'anios_desarrollo': list(ANIOS),
        'metricas_promedio': resumen.to_dict(orient='records'), 'variante_seleccionada': elegido,
        'comparacion_contra_D': {'D': tabla.loc['D'].to_dict(), 'seleccionada': tabla.loc[elegido].to_dict()},
        'criterio_seleccion': CRITERIO, 'hiperparametros_sin_modificar': algoritmo.get_params(),
        'pesos': 'balanceados calculados exclusivamente en train; pipeline XGBoost base existente',
        'ausentes': 'NaN sin imputar; ventanas nuevas completas; tratamiento nativo XGBoost',
        'proporcion_alto_bajo_denominador': 'casos Alto reales del fold',
        'plan_folds': plan.to_dict(orient='records'), 'procedencia': procedencia or {},
        'evaluacion_2025': None, '2025_participo_en_seleccion': False,
        'es_modelo_final_produccion': False, 'limitaciones': LIMITACIONES}
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino/ARCHIVOS[2]
    opt._json(ruta, reporte, exclusivo=True)
    resultados.to_csv(destino/ARCHIVOS[0], index=False)
    resumen.to_csv(destino/ARCHIVOS[1], index=False)
    # Las etiquetas 2025 se validan por primera vez después de congelar selección.
    anios = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M').year
    final = df.loc[anios <= 2025].copy()
    folds_final, _ = bt.crear_folds_expansivos(final)
    fold = next((f for f in folds_final if f[0] == 2025), None)
    if fold is None:
        raise ValueError('2025 no elegible; selección conservada sin comprobación.')
    final = agregar_features_flujo(ef.agregar_features_candidatas(final))
    reporte['estado'] = 'evaluacion_2025_iniciada_no_repetir'
    opt._json(ruta, reporte)
    externos = [ajustar(final, fold, n, 'comprobacion_2025') for n in dict.fromkeys(['D', elegido])]
    reporte.update(estado='completado_sin_produccion', evaluacion_2025=externos)
    resultados = pd.concat([resultados, pd.DataFrame(externos)], ignore_index=True)
    resultados.to_csv(destino/ARCHIVOS[0], index=False)
    opt._json(ruta, reporte)
    return resultados, resumen, reporte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-plan', action='store_true', help='Validar sin ajustar ni escribir artefactos')
    args = parser.parse_args()
    root = bt.ROOT
    ruta = root/'data/processed/dataset_modelo_ipress.csv'
    contenido = ruta.read_bytes()
    huella = hashlib.sha256(contenido).hexdigest()
    metadata = json.loads((ruta.parent/'dataset_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset_sha256') != huella:
        raise ValueError('Dataset no coincide con su metadata.')
    df = pd.read_csv(io.BytesIO(contenido), dtype={'codigo_ipress': str})
    # Solo lectura para preservar todos los artefactos preexistentes.
    protegidos = [*root.joinpath('data').rglob('*.csv'), *root.joinpath('models').glob('*')]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protegidos if p.is_file()}
    try:
        if args.solo_plan:
            h, folds, _ = preparar_desarrollo(df)
            print(json.dumps({'variantes': VARIANTES, 'anios_desarrollo': [f[0] for f in folds],
                'ajustes_desarrollo': 25, 'max_ajustes_2025': 2, 'filas_dataset': len(df),
                'ausentes_pct_desarrollo': (h[FEATURES].isna().mean()*100).to_dict()}, ensure_ascii=False, indent=2))
        else:
            _, resumen, r = evaluar(df, root/'models', procedencia={'dataset_sha256': huella,
                'fuentes_sha256': hashes, 'definicion_target': metadata.get('definicion_target')})
            print(resumen[['modelo', *ORDEN, 'seleccionada']].to_string(index=False))
            print('Seleccionada:', r['variante_seleccionada'])
    finally:
        if any(hashlib.sha256((root/p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError('Cambió una fuente o un artefacto anterior durante el experimento.')


if __name__ == '__main__':
    main()
