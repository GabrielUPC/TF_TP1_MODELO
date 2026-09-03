"""Doce reglas y diagnóstico de calibración de XGBoost D; sin producción."""
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
            'seleccion_regla_decision.json', 'calibracion_probabilidades.csv')
ARCHIVOS_EXTENSION = ('resultados_reglas_extension_020.csv', 'resumen_reglas_extension_020.csv',
                      'seleccion_regla_extension_020.json', 'probabilidades_reglas_extension_020.npz')
TOLERANCIA_REPRODUCCION = 1e-10
ORDEN = ['proporcion_alto_bajo_promedio', 'tasa_falsos_negativos_alto_promedio',
         'recall_alto_promedio', 'f1_macro_promedio', 'balanced_accuracy_promedio']
CRITERIO = {'max_caida_f1_vs_D_argmax': MAX_CAIDA_F1, 'orden': ORDEN,
            'sentido': ['min', 'min', 'max', 'max', 'max'],
            'denominador_proporcion_alto_bajo': 'casos Alto reales del fold',
            'agregacion': 'media aritmética de los cinco años, con igual peso por año',
            'empate': 'preferir argmax; después orden fijo de reglas declaradas',
            'tolerancia_limite_f1': 1e-12,
            'mejora_seguridad_minima': .005,
            'caida_global_claramente_mayor': .01,
            'veto_balance': 'Si ambas reducciones de error son <0.005 y la caída de F1 o balanced accuracy es >0.01, no seleccionar.',
            'expansion_alto_riesgosa': 'Aumento >=0.10 absoluto de proporción predicha Alto frente a argmax; alerta, no exclusión automática.',
            'naturaleza_limites': 'Convenciones exploratorias fijadas antes de evaluar, no umbrales clínicos.'}


def reglas_candidatas():
    return [{'regla': BASE, 'tipo_regla': 'argmax', 'umbral': None},
            *[{'regla': f'alto_{u:.2f}', 'tipo_regla': 'alto', 'umbral': u}
              for u in (.25, .30, .35, .40, .45)],
            *[{'regla': f'proteccion_{u:.2f}', 'tipo_regla': 'proteccion', 'umbral': u}
              for u in (.20, .25, .30, .35)],
            *[{'regla': f'combinada_{u:.2f}_0.25', 'tipo_regla': 'combinada',
               'umbral': u, 'umbral_proteccion': .25} for u in (.35, .40)]]


def reglas_extension_020():
    """Solo las seis comparaciones pedidas; no amplía la grilla original."""
    originales = {r['regla']: r for r in reglas_candidatas()}
    return [originales[n] for n in [BASE, 'proteccion_0.20', 'alto_0.40', 'alto_0.35']] + [
        {'regla': f'combinada_{u:.2f}_0.20', 'tipo_regla': 'combinada',
         'umbral': u, 'umbral_proteccion': .20} for u in (.40, .35)]


def validar_probabilidades(probabilidades):
    p = np.asarray(probabilidades, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-7)):
        raise ValueError('Probabilidades inválidas: se requiere matriz N x 3, orden Bajo/Medio/Alto y suma 1.')
    return p


def aplicar_regla(probabilidades, regla):
    """No muta ni recalibra p. Empates argmax: primera clase en orden 0/1/2."""
    if regla not in reglas_candidatas() and regla not in reglas_extension_020():
        raise ValueError('Regla fuera de los espacios fijos autorizados.')
    p = validar_probabilidades(probabilidades)
    pred = p.argmax(axis=1)
    if regla['tipo_regla'] == 'alto':
        pred[p[:, 2] >= regla['umbral']] = 2
    elif regla['tipo_regla'] == 'proteccion':
        pred[(pred == 0) & (p[:, 2] >= regla['umbral'])] = 1
    elif regla['tipo_regla'] == 'combinada':
        alto = p[:, 2] >= regla['umbral']
        proteger = ~alto & (pred == 0) & (p[:, 2] >= regla['umbral_proteccion'])
        pred[proteger] = 1
        pred[alto] = 2
    return pred


def diagnosticar_calibracion(y_real, probabilidades, anio):
    """Evaluación de probabilidades originales, sin ajustar calibradores.

    Brier multiclase = media(sum_c (p_c-y_c)^2), rango [0,2], sin dividir por 3.
    Reliability OVR: diez bins fijos [0,.1), ... [.9,1]. ECE pondera por tamaño.
    """
    if anio not in ANIOS_DESARROLLO:
        raise ValueError('Diagnóstico de calibración exclusivamente en desarrollo.')
    p = validar_probabilidades(probabilidades)
    y = bt._clases(y_real)
    if len(y) != len(p):
        raise ValueError('Probabilidades y etiquetas deben cubrir el mismo test.')
    observado = np.eye(3)[y]
    resumen = {'anio_prueba': anio, 'n_test': len(y),
        'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.)).mean()),
        'brier_multiclase': float(np.square(p-observado).sum(axis=1).mean()),
        'confianza_media_maxima': float(p.max(axis=1).mean()),
        'accuracy_argmax': float((p.argmax(axis=1) == y).mean())}
    resumen['brecha_confianza_accuracy'] = resumen['confianza_media_maxima']-resumen['accuracy_argmax']
    filas, eces = [], []
    for clase in range(3):
        bins = np.minimum((p[:, clase]*10).astype(int), 9)
        detalle, ece = [], 0.
        for b in range(10):
            mascara = bins == b
            n = int(mascara.sum())
            media = float(p[mascara, clase].mean()) if n else np.nan
            frecuencia = float(observado[mascara, clase].mean()) if n else np.nan
            if n:
                ece += n/len(y)*abs(media-frecuencia)
            detalle.append({'clase': clase, 'bin': b, 'limite_inferior': b/10,
                'limite_superior': (b+1)/10, 'ultimo_bin_incluye_1': b == 9,
                'n_bin': n, 'probabilidad_media': media, 'frecuencia_observada': frecuencia})
        resumen[f'ece_clase_{clase}'] = float(ece)
        resumen[f'brier_clase_{clase}'] = float(np.square(p[:, clase]-observado[:, clase]).mean())
        eces.append(ece)
        filas.extend({**resumen, **f, 'ece_clase': ece,
                      'brier_clase': resumen[f'brier_clase_{clase}']} for f in detalle)
    resumen['ece_macro'] = float(np.mean(eces))
    # Convención diagnóstica: no certificado de calibración ni selección de calibrador.
    resumen['alerta_calibracion'] = bool(len(y) >= 100 and
        (max(eces) >= .10 or abs(resumen['brecha_confianza_accuracy']) >= .10))
    resumen['observacion'] = ('Desajuste relevante según criterio exploratorio; revisar reliability antes de uso operativo.'
        if resumen['alerta_calibracion'] else
        'Sin alerta según criterio exploratorio; no demuestra probabilidades calibradas, especialmente con muestras pequeñas.')
    tabla = pd.DataFrame(filas)
    for c, valor in resumen.items():
        tabla[c] = valor
    return resumen, tabla


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
        pred = aplicar_regla(p, regla)
        metricas = bt.calcular_metricas_backtesting(df.loc[test, bt.OBJETIVO], pred, p)
        fila = opt._registro(df, test, train, regla['regla'], anio, metricas)
        fila['regla'] = fila.pop('configuracion')
        fila.update(tipo='regla_decision', tipo_regla=regla['tipo_regla'], umbral=regla['umbral'],
                    umbral_proteccion=regla.get('umbral_proteccion'), fase=fase, probabilidades_sha256=huella)
        for clase, nombre in enumerate(['bajo', 'medio', 'alto']):
            fila[f'predichos_{nombre}'] = int((pred == clase).sum())
            fila[f'proporcion_predicha_{nombre}'] = float((pred == clase).mean())
        proporcion_base_alto = float((p.argmax(axis=1) == 2).mean())
        fila['delta_proporcion_predicha_alto_vs_argmax'] = fila['proporcion_predicha_alto']-proporcion_base_alto
        fila['riesgo_expansion_alto'] = fila['delta_proporcion_predicha_alto_vs_argmax'] >= .10-1e-12
        filas.append(fila)
    return pd.DataFrame(filas)


def seleccionar_regla(resultados, *, reglas=None):
    reglas = reglas_candidatas() if reglas is None else reglas
    if reglas != reglas_candidatas() and reglas != reglas_extension_020():
        raise ValueError('Solo se permiten la grilla original o las seis reglas de extensión.')
    nombres = [r['regla'] for r in reglas]
    if resultados.empty or set(resultados.regla) != set(nombres):
        raise ValueError('La comparación requiere exactamente las reglas del espacio elegido, incluido argmax.')
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
    for nombre in ['bajo', 'medio', 'alto']:
        resumen[f'predichos_{nombre}_total'] = resumen.regla.map(resultados.groupby('regla')[f'predichos_{nombre}'].sum())
        resumen[f'proporcion_predicha_{nombre}_promedio'] = resumen.regla.map(resultados.groupby('regla')[f'proporcion_predicha_{nombre}'].mean())
    resumen['riesgo_expansion_alto_algun_anio'] = resumen.regla.map(resultados.groupby('regla').riesgo_expansion_alto.any())
    if not np.isfinite(resumen[ORDEN].to_numpy()).all():
        raise ValueError('Faltan métricas válidas para seleccionar.')
    referencia = resumen.set_index('regla').loc[BASE]
    for metrica in ORDEN:
        resumen['delta_' + metrica] = resumen[metrica] - referencia[metrica]
    resumen['admisible'] = resumen.f1_macro_promedio.ge(referencia.f1_macro_promedio-MAX_CAIDA_F1-1e-12)
    mejora_severa = -resumen.delta_proporcion_alto_bajo_promedio
    mejora_fnr = -resumen.delta_tasa_falsos_negativos_alto_promedio
    caida_global = np.maximum(-resumen.delta_f1_macro_promedio, -resumen.delta_balanced_accuracy_promedio)
    resumen['veto_mejora_minima_caida_global'] = (mejora_severa.lt(.005-1e-12)
        & mejora_fnr.lt(.005-1e-12) & caida_global.gt(.01+1e-12))
    resumen['mejora_seguridad'] = mejora_severa.gt(1e-12) | mejora_fnr.gt(1e-12)
    resumen['elegible_seleccion'] = (resumen.admisible & ~resumen.veto_mejora_minima_caida_global
        & (resumen.mejora_seguridad | resumen.regla.eq(BASE)))
    resumen['delta_proporcion_predicha_alto_promedio'] = resumen.proporcion_predicha_alto_promedio-referencia.proporcion_predicha_alto_promedio
    resumen['riesgo_expansion_alto_promedio'] = resumen.delta_proporcion_predicha_alto_promedio.ge(.10-1e-12)
    resumen['motivo_no_seleccionable'] = np.select([~resumen.admisible, resumen.veto_mejora_minima_caida_global,
        ~resumen.mejora_seguridad & resumen.regla.ne(BASE)],
        ['Caída F1 >0.02', 'Mejora seguridad <0.005 con caída global >0.01', 'Sin mejora de seguridad frente a argmax'], default='')
    indices = resumen.loc[resumen.elegible_seleccion].assign(_orden=lambda d: d.regla.map({r: i for i, r in enumerate(nombres)}))
    indices = indices.sort_values([*ORDEN, '_orden'], ascending=[True, True, False, False, False, True], kind='stable').index
    resumen['ranking'] = pd.Series({i: pos+1 for pos, i in enumerate(indices)}, dtype='Int64')
    elegido = resumen.loc[indices[0], 'regla']
    resumen['seleccionada'] = resumen.regla.eq(elegido)
    return resumen, elegido


def evaluar_extension_020(df, probabilidades_por_anio, referencia):
    """Compara desarrollo sin fit/predict ni lectura de 2025.

    Recibe matrices originales en el orden del test anterior. Verifica sus hashes
    y los del test contra resultados_reglas_decision.csv. Si no se conservaron
    las matrices, no es posible reconstruirlas a partir del resumen.
    """
    if set(probabilidades_por_anio) != set(ANIOS_DESARROLLO):
        raise ValueError('Se requieren matrices de los cinco años de desarrollo; no 2025.')
    historico, folds, _ = ef.preparar_desarrollo(df)
    filas = []
    for anio, train, test in folds:
        p = validar_probabilidades(probabilidades_por_anio[anio])
        huella = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
        ref = referencia.loc[referencia.regla.eq(BASE) & referencia.anio_prueba.eq(anio)
                             & referencia.fase.eq('desarrollo')]
        if len(ref) != 1:
            raise ValueError(f'Falta referencia argmax única de desarrollo {anio}.')
        ref = ref.iloc[0]
        registro = opt._registro(historico, test, train, BASE, anio, {})
        if huella != ref.probabilidades_sha256 or registro['test_sha256'] != ref.test_sha256:
            raise ValueError(f'Las probabilidades o el test {anio} no coinciden con el experimento original.')
        filas.append(evaluar_probabilidades(historico, train, test, p, anio,
                                            reglas_extension_020(), 'desarrollo'))
    resultados = pd.concat(filas, ignore_index=True)
    resumen, elegida = seleccionar_regla(resultados, reglas=reglas_extension_020())
    resumen['reduce_ambos_errores'] = (resumen.delta_tasa_falsos_negativos_alto_promedio.lt(-1e-12)
        & resumen.delta_proporcion_alto_bajo_promedio.lt(-1e-12))
    # Se informa el delta exacto de F1; no se inventa otro umbral de estabilidad.
    return resultados, resumen, elegida


def comprobar_extension_020_2025(df, probabilidades, referencia, regla_seleccionada):
    """Solo comprobación posterior a selección histórica; no ajusta modelos."""
    reglas = {r['regla']: r for r in reglas_extension_020()}
    if regla_seleccionada not in reglas:
        raise ValueError('La regla seleccionada no pertenece a la extensión.')
    periodos = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M')
    final = df.loc[periodos.year <= 2025].copy()
    folds, _ = bt.crear_folds_expansivos(final)
    fold = next((f for f in folds if f[0] == 2025), None)
    if fold is None:
        raise ValueError('2025 no es elegible.')
    final = agregar_features_candidatas(final)
    anio, train, test = fold
    p = validar_probabilidades(probabilidades)
    huella = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
    ref = referencia.loc[referencia.regla.eq(BASE) & referencia.anio_prueba.eq(2025)
                         & referencia.fase.eq('comprobacion_2025')]
    if len(ref) != 1 or huella != ref.iloc[0].probabilidades_sha256:
        raise ValueError('No coincide la matriz original de 2025.')
    if opt._registro(final, test, train, BASE, anio, {})['test_sha256'] != ref.iloc[0].test_sha256:
        raise ValueError('No coincide el test original de 2025.')
    comparadas = [reglas[n] for n in dict.fromkeys([BASE, regla_seleccionada])]
    return evaluar_probabilidades(final, train, test, p, anio, comparadas, 'comprobacion_2025')


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
    filas, calibracion, tablas_calibracion = [], [], []
    for anio, train, test in folds:
        print(f'XGBoost D {anio}: un ajuste, calibración y doce decisiones sobre el mismo test', flush=True)
        p = probabilidades_fold(motor, algoritmo, historico, train, test, columnas, anio)
        diagnostico, tabla_calibracion = diagnosticar_calibracion(historico.loc[test, bt.OBJETIVO], p, anio)
        calibracion.append(diagnostico)
        tabla_calibracion['test_sha256'] = opt._registro(historico, test, train, BASE, anio, {})['test_sha256']
        tabla_calibracion['probabilidades_sha256'] = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
        tablas_calibracion.append(tabla_calibracion)
        filas.append(evaluar_probabilidades(historico, train, test, p, anio, reglas, 'desarrollo'))
    resultados = pd.concat(filas, ignore_index=True)
    resumen, nombre = seleccionar_regla(resultados)
    elegida = next(r for r in reglas if r['regla'] == nombre)
    tabla = resumen.set_index('regla')
    reporte = {'version_experimento': 'reglas_decision_D_calibracion_12_v3',
        'estado': 'seleccion_historica_congelada_antes_de_2025',
        'reglas_probadas': reglas, 'anios_desarrollo': list(ANIOS_DESARROLLO),
        'regla_base': reglas[0],
        'reglas_admisibles': resumen.loc[resumen.admisible, 'regla'].tolist(),
        'reglas_no_admisibles': resumen.loc[~resumen.admisible, 'regla'].tolist(),
        'reglas_elegibles_seleccion': resumen.loc[resumen.elegible_seleccion, 'regla'].tolist(),
        'calibracion_por_anio': calibracion,
        'observaciones_calibracion': {
            'metodo': 'Diagnóstico sin calibrador, antes de aplicar reglas. Brier=sum cuadrática de las 3 clases, promedio por caso [0,2]. Log-loss con clip solo para log a 1e-15.',
            'reliability': '10 bins de ancho 0.1, OVR por clase; ECE ponderado por n_bin/n_test; bins vacíos NaN.',
            'alerta_exploratoria': 'Con n>=100: ECE de alguna clase >=0.10 o brecha absoluta confianza máxima-accuracy >=0.10. No es umbral clínico ni prueba estadística.',
            'anios_con_alerta': [c['anio_prueba'] for c in calibracion if c['alerta_calibracion']],
            'calibrador_ajustado': False},
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
    pd.concat(tablas_calibracion, ignore_index=True).to_csv(destino / ARCHIVOS[3], index=False)

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


def ejecutar_extension_020(df, destino, *, dataset_sha256, motor=None, archivo_probabilidades=None):
    """Ruta aislada: reutiliza matrices verificadas o reproduce una vez por fold.

    No modifica el experimento original. Una divergencia aborta sin relajar
    tolerancias, reintentar fits ni abrir el año 2025 antes de la selección.
    """
    destino = Path(destino)
    if any((destino/n).exists() for n in ARCHIVOS_EXTENSION):
        raise FileExistsError('Ya existe la extensión; revisar su estado, no sobrescribir ni repetir 2025.')
    referencia = pd.read_csv(destino/ARCHIVOS[0])
    original = json.loads((destino/ARCHIVOS[2]).read_text(encoding='utf-8'))
    if (original.get('procedencia', {}).get('dataset_sha256') != dataset_sha256
            or original.get('anios_desarrollo') != list(ANIOS_DESARROLLO)
            or original.get('conjunto_features') != 'D'):
        raise ValueError('Dataset, años o features de referencia no coinciden con el experimento D.')
    historico, folds, _ = ef.preparar_desarrollo(df)
    columnas = original['columnas_predictoras']
    if len(columnas) != len(set(columnas)) or set(columnas) & bt.ETIQUETAS:
        raise ValueError('Predictores de referencia inválidos.')
    archivo = Path(archivo_probabilidades) if archivo_probabilidades is not None else destino/'probabilidades_reglas_decision.npz'
    if archivo_probabilidades is not None and not archivo.is_file():
        raise FileNotFoundError('No existe el archivo de probabilidades indicado.')
    disponibles = set()
    if archivo.is_file():
        with np.load(archivo, allow_pickle=False) as guardadas:
            disponibles = set(guardadas.files)  # No leer la matriz 2025 todavía.
    algoritmo = None
    def obtener_motor():
        nonlocal motor, algoritmo
        if algoritmo is not None:
            return
        motor = motor if motor is not None else bt._motor_existente()
        if [*motor.COLUMNAS_PREDICTORAS, *CONJUNTOS['D']] != columnas:
            raise ValueError('Las features actuales difieren de la referencia D.')
        algoritmo = motor.obtener_modelos().get('XGBoost')
        if algoritmo is None:
            raise RuntimeError('XGBoost no disponible.')
        parametros = json.loads(pd.Series({'p': algoritmo.get_params()}).to_json())['p']
        if parametros != original['hiperparametros_sin_modificar']:
            raise ValueError('La configuración XGBoost no coincide con la original.')
    # Fallar antes de crear salidas si falta el entorno necesario para desarrollo.
    if any(f'p_{a}' not in disponibles for a in ANIOS_DESARROLLO):
        obtener_motor()
    reporte = {'version': 'extension_020_cli_v1', 'estado': 'desarrollo_iniciado',
        'reglas_probadas': reglas_extension_020(), 'anios_desarrollo': list(ANIOS_DESARROLLO),
        'criterio_seleccion': CRITERIO, 'tolerancia_argmax': {'atol': TOLERANCIA_REPRODUCCION, 'rtol': 0},
        'procedencia': {'dataset_sha256': dataset_sha256,
            'resultados_originales_sha256': hashlib.sha256((destino/ARCHIVOS[0]).read_bytes()).hexdigest(),
            'seleccion_original_sha256': hashlib.sha256((destino/ARCHIVOS[2]).read_bytes()).hexdigest(),
            'archivo_probabilidades': str(archivo) if archivo.is_file() else None,
            'archivo_probabilidades_sha256': hashlib.sha256(archivo.read_bytes()).hexdigest() if archivo.is_file() else None},
        'columnas_predictoras': columnas, 'hiperparametros': original['hiperparametros_sin_modificar'],
        'verificacion_por_fold': [], 'regla_seleccionada': None, 'evaluacion_2025': None,
        '2025_participo_en_seleccion': False, 'es_modelo_final_produccion': False,
        'advertencia': '2025 ya observado: comprobación adicional, no holdout virgen. Reproducidas no significa reutilizadas; métricas iguales no prueban identidad de matrices.'}
    ruta = destino/ARCHIVOS_EXTENSION[2]
    opt._json(ruta, reporte, exclusivo=True)  # Reserva exclusiva antes del primer fit.
    matrices = {}
    def obtener_verificar(datos, fold, fase):
        anio, train, test = fold
        ref = referencia.loc[referencia.regla.eq(BASE) & referencia.anio_prueba.eq(anio) & referencia.fase.eq(fase)]
        if len(ref) != 1:
            raise ValueError(f'Falta referencia argmax única para {anio}.')
        ref = ref.iloc[0]
        registro = opt._registro(datos, test, train, BASE, anio, {})
        if any(registro[c] != ref[c] for c in ['test_sha256', 'n_train', 'n_test']):
            raise ValueError(f'No coinciden test o tamaños train/test de {anio}.')
        if f'p_{anio}' in disponibles:
            with np.load(archivo, allow_pickle=False) as guardadas:
                p = validar_probabilidades(guardadas[f'p_{anio}']).copy()
            origen = 'reutilizadas'
        else:
            obtener_motor()
            print(f'Extensión 020 {anio}: reproducir un único ajuste XGBoost D', flush=True)
            p = probabilidades_fold(motor, algoritmo, datos, train, test, columnas, anio)
            origen = 'reproducidas'
        huella = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
        igual = huella == ref.probabilidades_sha256
        if origen == 'reutilizadas' and not igual:
            raise ValueError(f'La matriz persistida {anio} no coincide con el hash original.')
        actual = evaluar_probabilidades(datos, train, test, p, anio, [reglas_extension_020()[0]], fase).iloc[0]
        campos = [*bt.METRICAS, *[f'predichos_{c}' for c in ['bajo', 'medio', 'alto']],
                  *[f'proporcion_predicha_{c}' for c in ['bajo', 'medio', 'alto']]]
        diferencias = {c: float(actual[c])-float(ref[c]) for c in campos}
        if any(not np.isclose(float(actual[c]), float(ref[c]), atol=TOLERANCIA_REPRODUCCION,
                              rtol=0, equal_nan=True) for c in campos):
            raise ValueError(f'Argmax {anio} no reproduce la referencia con tolerancia 1e-10; no continuar.')
        reporte['verificacion_por_fold'].append({'anio': anio, 'origen_probabilidades': origen,
            'n_ajustes': int(origen == 'reproducidas'), 'test_sha256': registro['test_sha256'],
            'probabilidades_sha256': huella, 'hash_original_probabilidades': ref.probabilidades_sha256,
            'matriz_identica_por_hash': bool(igual), 'argmax_reproducido': True, 'deltas_argmax': diferencias})
        matrices[f'p_{anio}'] = p
        return p
    try:
        filas = []
        for fold in folds:
            p = obtener_verificar(historico, fold, 'desarrollo')
            anio, train, test = fold
            filas.append(evaluar_probabilidades(historico, train, test, p, anio, reglas_extension_020(), 'desarrollo'))
        resultados = pd.concat(filas, ignore_index=True)
        resumen, elegida = seleccionar_regla(resultados, reglas=reglas_extension_020())
        resumen['reduce_ambos_errores'] = (resumen.delta_tasa_falsos_negativos_alto_promedio.lt(-1e-12)
            & resumen.delta_proporcion_alto_bajo_promedio.lt(-1e-12))
        reporte.update(estado='seleccion_congelada_antes_de_2025', regla_seleccionada=elegida,
            metricas_promedio=resumen.to_dict(orient='records'))
        opt._json(ruta, reporte)
        resultados.to_csv(destino/ARCHIVOS_EXTENSION[0], index=False)
        resumen.to_csv(destino/ARCHIVOS_EXTENSION[1], index=False)
        np.savez_compressed(destino/ARCHIVOS_EXTENSION[3], **matrices)
        # 2025 solo se construye/obtiene después de guardar la selección histórica.
        anios = pd.PeriodIndex(df.periodo_predicho.astype(str), freq='M').year
        final = df.loc[anios <= 2025].copy()
        finales, _ = bt.crear_folds_expansivos(final)
        fold = next((f for f in finales if f[0] == 2025), None)
        if fold is None:
            raise ValueError('2025 no elegible; selección histórica conservada.')
        final = agregar_features_candidatas(final)
        reporte['estado'] = 'comprobacion_2025_iniciada_no_repetir'
        opt._json(ruta, reporte)
        p = obtener_verificar(final, fold, 'comprobacion_2025')
        anio, train, test = fold
        reglas = [r for r in reglas_extension_020() if r['regla'] in {BASE, elegida}]
        externas = evaluar_probabilidades(final, train, test, p, anio, reglas, 'comprobacion_2025')
        reporte.update(estado='completado_sin_produccion', evaluacion_2025=externas.to_dict(orient='records'))
        resultados = pd.concat([resultados, externas], ignore_index=True)
        resultados.to_csv(destino/ARCHIVOS_EXTENSION[0], index=False)
        np.savez_compressed(destino/ARCHIVOS_EXTENSION[3], **matrices)
        opt._json(ruta, reporte)
        return resultados, resumen, reporte
    except Exception as error:
        reporte.update(estado='fallido_no_reintentar_sin_revision', error=str(error))
        opt._json(ruta, reporte)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-plan', action='store_true', help='Verificar años sin ajustar modelos ni escribir resultados')
    parser.add_argument('--extension-020', action='store_true', help='Solo seis reglas; salidas separadas y reproducción verificada si faltan matrices')
    parser.add_argument('--probabilidades-originales', type=Path, help='NPZ opcional con p_2018,...,p_2025; requiere --extension-020')
    args = parser.parse_args()
    if args.probabilidades_originales is not None and not args.extension_020:
        parser.error('--probabilidades-originales requiere --extension-020')
    ruta = bt.ROOT / 'data/processed/dataset_modelo_ipress.csv'
    contenido = ruta.read_bytes()
    huella = hashlib.sha256(contenido).hexdigest()
    metadata = json.loads((ruta.parent / 'dataset_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset_sha256') != huella:
        raise ValueError('El dataset no coincide con su metadata.')
    df = pd.read_csv(io.BytesIO(contenido), dtype={'codigo_ipress': str})
    if args.solo_plan:
        _, folds, _ = ef.preparar_desarrollo(df)
        print(json.dumps({'reglas': reglas_extension_020() if args.extension_020 else reglas_candidatas(), 'anios_desarrollo': [f[0] for f in folds],
            'comprobacion_adicional': 2025,
            ('max_ajustes_si_reproducir' if args.extension_020 else 'ajustes_totales_previstos'): len(folds)+1,
            'conjunto': 'D', 'dataset_sha256': huella}, ensure_ascii=False, indent=2))
    elif args.extension_020:
        protegidos = [ruta, ruta.parent/'dataset_metadata.json',
            *[bt.ROOT/'models'/n for n in ARCHIVOS], *bt.ROOT.joinpath('models').glob('*.joblib')]
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protegidos}
        try:
            _, resumen, reporte = ejecutar_extension_020(df, bt.ROOT/'models', dataset_sha256=huella,
                archivo_probabilidades=args.probabilidades_originales)
            print(resumen.to_string(index=False))
            print(json.dumps({'regla_seleccionada': reporte['regla_seleccionada'],
                              'evaluacion_2025': reporte['evaluacion_2025']}, ensure_ascii=False, default=str))
        finally:
            if any(hashlib.sha256(p.read_bytes()).hexdigest() != h for p, h in hashes.items()):
                raise RuntimeError('Cambió un archivo original protegido durante la extensión.')
    else:
        _, resumen, reporte = evaluar_reglas(df, bt.ROOT / 'models', procedencia={'dataset_sha256': huella})
        print(resumen[['regla', *ORDEN, 'admisible', 'ranking']].to_string(index=False))
        print(json.dumps({'regla_seleccionada': reporte['regla_seleccionada'],
                          'evaluacion_2025': reporte['evaluacion_2025']}, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
