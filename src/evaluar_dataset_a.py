"""Cinco variantes D/A con XGBoost base y evaluación temporal, sin producción."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtesting_temporal as bt
from . import evaluar_features_temporales as ef
from . import optimizar_xgboost as opt
from . import preparar_dataset_a as pa

ANIOS = (2018, 2021, 2022, 2023, 2024)
VARIANTES = {'D': [], 'D+A_CAMAS': ['CA_CAMAS'],
    'D+A_RECURSOS': list(pa.VARIABLES),
    'D+A_RATIOS': [*pa.VARIABLES, *pa.RATIOS],
    'D+A_COMPLETO': list(pa.CANDIDATAS)}
ABLACION = ('CA_CAMAS', 'CA_MEDICOS_TOTAL', 'CA_ENFERMERAS',
            'medicos_por_cama', 'enfermeras_por_cama')
ORDEN = [*bt.ORDEN_COMPARACION, 'proporcion_alto_bajo_promedio']
ARCHIVOS = ('resultados_dataset_a.csv', 'resumen_dataset_a.csv', 'seleccion_dataset_a.json')
CRITERIO = {'orden': ORDEN, 'sentido': ['max', 'max', 'max', 'min', 'min'],
    'max_caida_f1_vs_D': .02, 'agregacion': 'Media aritmética por año, igual peso para los cinco años',
    'empate': 'Orden fijo D, D+A_CAMAS, D+A_RECURSOS, D+A_RATIOS, D+A_COMPLETO',
    'ablacion': 'Diagnóstico histórico posterior; no cambia selección ni decisión de evaluar 2025',
    'tolerancia_limite_f1': 1e-12}


def validar_candidato(d, a, experimental, ambiguas):
    """Verifica el join guardado y la continuidad A; no reconstruye archivos fuente."""
    if not set([*pa.CLAVE, *pa.CANDIDATAS, 'estado_calidad_a']).issubset(a):
        raise ValueError('A analítico incompleto.')
    if not a.estado_calidad_a.isin(['UNICA', 'DUPLICADO_EXACTO']).all():
        raise ValueError('A contiene un estado de calidad no utilizable.')
    if a[pa.CLAVE].isna().any().any() or a.duplicated(pa.CLAVE).any():
        raise ValueError('A tiene claves inválidas o duplicadas.')
    if a.merge(ambiguas[pa.CLAVE].drop_duplicates(), on=pa.CLAVE, how='inner').shape[0]:
        raise ValueError('Una clave ambigua aporta covariables A.')
    if np.isinf(a[pa.CANDIDATAS].to_numpy(dtype=float)).any() or a[pa.VARIABLES].lt(0).any().any():
        raise ValueError('A tiene recursos negativos o infinitos.')
    recalculado = pa.agregar_derivadas(a[[*pa.CLAVE, *pa.VARIABLES, 'estado_calidad_a']])
    original = a.sort_values(pa.CLAVE).reset_index(drop=True)
    pd.testing.assert_frame_equal(original[pa.CANDIDATAS], recalculado[pa.CANDIDATAS],
                                  check_dtype=False, rtol=1e-10, atol=1e-12)
    esperado = pa.unir_d_a(d, a)
    if set(experimental) != set(esperado) or len(experimental) != len(d):
        raise ValueError('El experimental no conserva exactamente las filas/columnas esperadas.')
    pd.testing.assert_frame_equal(experimental[d.columns].reset_index(drop=True), d.reset_index(drop=True))
    pd.testing.assert_frame_equal(experimental[esperado.columns].reset_index(drop=True), esperado,
                                  check_dtype=False, rtol=1e-10, atol=1e-12)
    return experimental.copy()


def cargar_datos(root=bt.ROOT):
    root = Path(root)
    procesado = root/'data/processed'
    calidad = root/'data/quality/dataset_a'
    paths = [procesado/'dataset_modelo_ipress.csv', procesado/'dataset_a_analitico.csv',
        procesado/'dataset_modelo_ipress_con_a_experimental.csv', calidad/'claves_ambiguas_a.csv',
        procesado/'dataset_metadata.json', calidad/'resumen_preparacion_dataset_a.json']
    protegidos = [*paths, *sorted((root/'data/raw').rglob('*.csv')), *sorted((root/'models').rglob('*.joblib'))]
    hashes = {str(p.relative_to(root)): pa.sha256(p) for p in protegidos}
    metadata = json.loads(paths[4].read_text(encoding='utf-8'))
    if metadata.get('dataset_sha256') != pa.sha256(paths[0]):
        raise ValueError('D no coincide con dataset_metadata.json.')
    preparacion = json.loads(paths[5].read_text(encoding='utf-8'))
    for nombre, huella in preparacion['fuentes_sha256'].items():
        fuente = (root/nombre).resolve()
        if not fuente.is_relative_to(root.resolve()) or pa.sha256(fuente) != huella:
            raise ValueError('Una fuente cambió desde la preparación de A.')
    d, a, experimental, ambiguas = [pd.read_csv(p, dtype={'codigo_ipress': str}) for p in paths[:4]]
    df = validar_candidato(d, a, experimental, ambiguas)
    return df, {'fuentes_sha256': hashes, 'definicion_target': metadata.get('definicion_target'),
                'version_preparacion_a': preparacion.get('version')}


def seleccionar(resultados):
    if set(resultados.modelo) != set(VARIANTES) or not resultados.fase.eq('desarrollo').all():
        raise ValueError('Se requieren exactamente las cinco variantes en desarrollo.')
    for _, g in resultados.groupby('modelo'):
        if set(g.anio_prueba) != set(ANIOS) or len(g) != len(ANIOS):
            raise ValueError('Solo los cinco años de desarrollo; nunca 2025 ni filas duplicadas.')
    for _, g in resultados.groupby('anio_prueba'):
        if g.test_sha256.isna().any() or g.test_sha256.nunique() != 1 or g.n_test.nunique() != 1:
            raise ValueError('Las variantes no comparten exactamente el test.')
    resumen = bt.resumir_backtesting(resultados)
    if not np.isfinite(resumen[ORDEN].to_numpy()).all():
        raise ValueError('Métricas de selección no evaluables.')
    base = resumen.set_index('modelo').loc['D']
    for m in ORDEN:
        resumen['delta_' + m] = resumen[m] - base[m]
    resumen['admisible'] = resumen.f1_macro_promedio.ge(base.f1_macro_promedio-.02-1e-12)
    candidatos = resumen.loc[resumen.admisible].assign(
        _orden=lambda x: x.modelo.map({v: i for i, v in enumerate(VARIANTES)}))
    indices = candidatos.sort_values([*ORDEN, '_orden'],
        ascending=[False, False, False, True, True, True], kind='stable').index
    ganador = resumen.loc[indices[0], 'modelo']
    resumen['ranking'] = pd.Series({i: n+1 for n, i in enumerate(indices)}, dtype='Int64')
    resumen['seleccionada'] = resumen.modelo.eq(ganador)
    return resumen, ganador


def cobertura(df):
    filas = []
    objetivo = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    for nombre, mascara in [('TOTAL', np.ones(len(df), dtype=bool)),
                            *[(str(anio), objetivo.year == anio) for anio in [*ANIOS, 2025]]]:
        g = df.loc[mascara]
        filas.append({'periodo_objetivo': nombre, 'n': len(g),
            'con_a_valida': int(g.tiene_datos_a.eq(1).sum()),
            'sin_a_valida': int(g.tiene_datos_a.eq(0).sum()),
            'cobertura_pct': float(g.tiene_datos_a.eq(1).mean()*100),
            'ausentes_pct_por_variable': (g[pa.CANDIDATAS].isna().mean()*100).to_dict()})
    return filas


def evaluar(df, destino, *, motor=None, procedencia=None):
    destino = Path(destino)
    if any((destino/n).exists() for n in ARCHIVOS):
        raise FileExistsError('El experimento ya existe; no sobrescribir ni repetir 2025 automáticamente.')
    if not set([*pa.CANDIDATAS, 'tiene_datos_a']).issubset(df):
        raise ValueError('Faltan covariables A.')
    if not df.tiene_datos_a.isin([0, 1]).all() or df.loc[df.tiene_datos_a.eq(0), pa.CANDIDATAS].notna().any().any():
        raise ValueError('Filas sin A válida no deben aportar covariables.')
    historico, folds, plan = ef.preparar_desarrollo(df)
    motor = motor if motor is not None else bt._motor_existente()
    columnas_d = [*motor.COLUMNAS_PREDICTORAS, *ef.CONJUNTOS['D']]
    columnas = {nombre: [*columnas_d, *extras] for nombre, extras in VARIANTES.items()}
    for cols in columnas.values():
        if len(cols) != len(set(cols)) or set(cols) & (bt.ETIQUETAS | set(motor.COLUMNAS_EXCLUIDAS)):
            raise ValueError('Predictores repetidos o etiquetas en el conjunto D/A.')
    algoritmo = motor.obtener_modelos().get('XGBoost')
    if algoritmo is None:
        raise RuntimeError('XGBoost no disponible.')
    filas = []
    def ajustar(datos, fold, nombre, cols, fase):
        anio, train, test = fold
        print(f'Dataset A {fase} {anio}: {nombre}', flush=True)
        _, metricas = ef._ajustar_evaluar(motor, algoritmo, datos, train, test, cols, anio)
        return ef._registro(datos, train, test, nombre, anio, metricas, cols, fase)
    for fold in folds:
        for nombre, cols in columnas.items():
            filas.append(ajustar(historico, fold, nombre, cols, 'desarrollo'))
    resultados = pd.DataFrame(filas)
    resumen, ganador = seleccionar(resultados)
    tabla = resumen.set_index('modelo')
    reporte = {'version': 'evaluacion_dataset_a_v1', 'estado': 'seleccion_congelada_antes_de_ablacion_y_2025',
        'variantes_evaluadas': list(VARIANTES), 'variables_por_variante': columnas,
        'anios_desarrollo': list(ANIOS), 'variante_seleccionada': ganador,
        'metricas_promedio': resumen.to_dict(orient='records'),
        'comparacion_contra_D': {'D': tabla.loc['D'].to_dict(), 'seleccionada': tabla.loc[ganador].to_dict()},
        'criterio_seleccion': CRITERIO, 'hiperparametros_fijos': algoritmo.get_params(),
        'pesos': 'balanceado calculado solo con y_train, igual que experimento D',
        'faltantes': 'NaN nativo XGBoost; sin imputación; escalador/one-hot ajustados solo en train',
        'tiene_datos_a_como_predictor': False, 'cobertura_a_usada': cobertura(df),
        'plan_folds': plan.to_dict(orient='records'), 'ablacion': [], 'evaluacion_2025': None,
        '2025_participo_en_seleccion': False, 'es_modelo_final_produccion': False,
        'procedencia': procedencia or {},
        'limitaciones': ['A representa contexto institucional, no personal/camas asignados a cada servicio.',
            'A(t) debe estar disponible al predecir t+1; fechas de publicación y revisiones no confirmadas.',
            '2025 ya observado anteriormente: no es un holdout completamente virgen.',
            'Años de desarrollo reutilizados: riesgo de sobreajuste por múltiples experimentos.',
            'El historial D procesado omite meses sin pareja objetivo; se conserva esa limitación.',
            'D conserva features temporales de riesgo observado del experimento previo; nunca etiqueta futura.',
            'Ablación individual conserva derivados correlacionados: no demuestra causalidad ni aporte independiente.']}
    destino.mkdir(parents=True, exist_ok=True)
    seleccion_path = destino/ARCHIVOS[2]
    opt._json(seleccion_path, reporte, exclusivo=True)
    resultados.to_csv(destino/ARCHIVOS[0], index=False)
    resumen.to_csv(destino/ARCHIVOS[1], index=False)
    ablaciones = []
    for quitar in ABLACION:
        if quitar not in VARIANTES[ganador]:
            continue
        reducidas = [c for c in columnas[ganador] if c != quitar]
        for fold in folds:
            fila = ajustar(historico, fold, f'{ganador}_sin_{quitar}', reducidas, 'ablacion')
            referencia = resultados.loc[resultados.modelo.eq(ganador) & resultados.anio_prueba.eq(fold[0])].iloc[0]
            fila['variable_retirada'] = quitar
            fila.update({f'delta_{m}_vs_ganadora': fila[m]-referencia[m] for m in bt.METRICAS})
            ablaciones.append(fila)
    reporte['ablacion'] = ablaciones
    reporte['ablacion_modifica_seleccion'] = False
    reporte['resumen_ablacion'] = bt.resumir_backtesting(pd.DataFrame(ablaciones)).to_dict(orient='records') if ablaciones else []
    filas.extend(ablaciones)
    pd.DataFrame(filas).to_csv(destino/ARCHIVOS[0], index=False)
    opt._json(seleccion_path, reporte)
    # Solo tras congelar y terminar ablación se validan etiquetas de 2025.
    anios = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M').year
    final = df.loc[anios <= 2025].copy()
    folds_final, _ = bt.crear_folds_expansivos(final)
    fold = next((f for f in folds_final if f[0] == 2025), None)
    if fold is None:
        raise ValueError('2025 no elegible; selección histórica conservada.')
    final = ef.agregar_features_candidatas(final)
    reporte['estado'] = 'evaluacion_2025_iniciada_no_repetir'
    opt._json(seleccion_path, reporte)
    externos = [ajustar(final, fold, nombre, columnas[nombre], 'evaluacion_2025')
                for nombre in dict.fromkeys(['D', ganador])]
    reporte.update(estado='completado_sin_produccion', evaluacion_2025=externos)
    filas.extend(externos)
    resultados = pd.DataFrame(filas)
    resultados.to_csv(destino/ARCHIVOS[0], index=False)
    opt._json(seleccion_path, reporte)
    return resultados, resumen, reporte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-plan', action='store_true', help='Valida datos/años sin ajustes ni escribir resultados')
    args = parser.parse_args()
    df, procedencia = cargar_datos()
    try:
        if args.solo_plan:
            _, folds, _ = ef.preparar_desarrollo(df)
            print(json.dumps({'anios_desarrollo': [f[0] for f in folds], 'variantes': VARIANTES,
                'ajustes_desarrollo': 25, 'max_ajustes_ablacion': 25, 'max_ajustes_2025': 2,
                'cobertura_a': cobertura(df)}, ensure_ascii=False, indent=2))
        else:
            _, resumen, reporte = evaluar(df, bt.ROOT/'models', procedencia=procedencia)
            print(resumen[['modelo', *ORDEN, 'seleccionada']].to_string(index=False))
            print('Seleccionada:', reporte['variante_seleccionada'])
    finally:
        if any(pa.sha256(bt.ROOT/p) != h for p, h in procedencia['fuentes_sha256'].items()):
            raise RuntimeError('Una fuente o modelo protegido cambió durante el experimento.')


if __name__ == '__main__':
    main()
