"""Variables candidatas aisladas, calculables al cierre de t; sin tocar producción."""
from __future__ import annotations

import numpy as np
import pandas as pd

GRUPOS = ['codigo_ipress', 'servicio_hospitalizacion']
LAGS = {
    **{f'ocupacion_lag_{k}m': ('ocupacion_estimada', k) for k in (1, 2, 3)},
    **{f'{nombre}_lag_{k}m': (fuente, k)
       for nombre, fuente in [('pacientes_camas', 'total_pacientes_camas'),
          ('camas_disponibles', 'total_camas_disponibles'),
          ('ingresos', 'total_ingresos'), ('egresos', 'total_egresos')]
       for k in (1, 2)},
}
# Las diferencias de ocupación/ingresos/egresos de 1m ya son tendencias BASE.
CAMBIOS = {
    'cambio_ocupacion_2m': ('ocupacion_estimada', 'ocupacion_lag_2m'),
    'cambio_pacientes_camas_1m': ('total_pacientes_camas', 'pacientes_camas_lag_1m'),
    'cambio_camas_disponibles_1m': ('total_camas_disponibles', 'camas_disponibles_lag_1m'),
}
HISTORIA = ['max_ocupacion_3m', 'min_ocupacion_3m', 'desviacion_ocupacion_3m',
            'promedio_ocupacion_6m', 'max_presion_3m',
            'promedio_pacientes_camas_3m', 'promedio_camas_disponibles_3m']
RIESGO = ['meses_alto_ultimos_3m', 'meses_alto_ultimos_6m',
          'meses_medio_alto_ultimos_3m', 'meses_consecutivos_alto']
MARGENES = ['margen_umbral_alto', 'margen_umbral_medio', 'distancia_absoluta_umbral_alto']
CRECIMIENTO = ['crecimiento_demanda_1m', 'crecimiento_capacidad_1m',
               'brecha_crecimiento_demanda_capacidad']
CONJUNTOS = {'Base': [], 'A': list(LAGS), 'B': [*LAGS, *CAMBIOS, 'aceleracion_ocupacion'],
             'C': [*LAGS, *CAMBIOS, 'aceleracion_ocupacion', *HISTORIA],
             'D': [*LAGS, *CAMBIOS, 'aceleracion_ocupacion', *HISTORIA, *RIESGO, *MARGENES, *CRECIMIENTO]}
NUEVAS_FEATURES = CONJUNTOS['D']
REUTILIZADAS = {
    'cambio_ocupacion_1m': 'tendencia_ocupacion_1m',
    'cambio_ingresos_1m': 'tendencia_ingresos_1m',
    'cambio_egresos_1m': 'tendencia_egresos_1m',
    'promedio_ocupacion_3m': 'promedio_movil_3m_ocupacion',
}
ALIAS_EXPERIMENTALES = {'max_presion_ingresos_3m': 'max_presion_3m'}


def definiciones_features():
    """Catálogo trazable: fórmulas, fuentes y conjuntos; no estadísticas globales."""
    definiciones = {}
    def registrar(nombre, formula, fuentes, grupo):
        definiciones[nombre] = {'definicion': nombre.replace('_', ' '), 'formula': formula,
            'variables_necesarias': fuentes, 'grupo': grupo,
            'conjuntos': [c for c, features in CONJUNTOS.items() if nombre in features]}
    for nombre, (fuente, k) in LAGS.items():
        registrar(nombre, f'{fuente}(t-{k}); exige todos los meses intermedios', [fuente], 'A')
    for nombre, (fuente, lag) in CAMBIOS.items():
        registrar(nombre, f'{fuente}(t) - {lag}', [fuente, lag], 'B')
    registrar('aceleracion_ocupacion', 'ocupacion(t) - 2*ocupacion(t-1) + ocupacion(t-2)',
              ['ocupacion_estimada', 'ocupacion_lag_1m', 'ocupacion_lag_2m'], 'B')
    for nombre, fuente, k, operacion in [
        ('max_ocupacion_3m', 'ocupacion_estimada', 3, 'max'),
        ('min_ocupacion_3m', 'ocupacion_estimada', 3, 'min'),
        ('desviacion_ocupacion_3m', 'ocupacion_estimada', 3, 'std(ddof=0)'),
        ('promedio_ocupacion_6m', 'ocupacion_estimada', 6, 'mean'),
        ('max_presion_3m', 'presion_ingresos_camas', 3, 'max'),
        ('promedio_pacientes_camas_3m', 'total_pacientes_camas', 3, 'mean'),
        ('promedio_camas_disponibles_3m', 'total_camas_disponibles', 3, 'mean')]:
        registrar(nombre, f'{operacion}({fuente}[t-{k-1}:t]); {k} meses completos consecutivos', [fuente], 'C')
    for nombre, k, condicion in [('meses_alto_ultimos_3m', 3, '=2'),
                               ('meses_alto_ultimos_6m', 6, '=2'),
                               ('meses_medio_alto_ultimos_3m', 3, '>=1')]:
        registrar(nombre, f'sum(riesgo_actual {condicion}) en t-{k-1}:t; ventana completa',
                  ['nivel_riesgo_actual_codificado'], 'D')
    registrar('meses_consecutivos_alto', 'racha hasta t de riesgo_actual=2; parar en hueco/no Alto/ausente',
              ['nivel_riesgo_actual_codificado'], 'D')
    for nombre, formula in [('margen_umbral_alto', 'ocupacion_estimada - 0.85'),
                           ('margen_umbral_medio', 'ocupacion_estimada - 0.70'),
                           ('distancia_absoluta_umbral_alto', 'abs(ocupacion_estimada - 0.85)')]:
        registrar(nombre, formula, ['ocupacion_estimada'], 'E')
    for nombre, fuente in [('crecimiento_demanda_1m', 'total_pacientes_camas'),
                          ('crecimiento_capacidad_1m', 'total_camas_disponibles')]:
        registrar(nombre, f'({fuente}(t)-{fuente}(t-1))/{fuente}(t-1); NaN si previo<=0 o actual<0', [fuente], 'F')
    registrar('brecha_crecimiento_demanda_capacidad', 'crecimiento_demanda_1m - crecimiento_capacidad_1m',
              CRECIMIENTO[:2], 'F')
    return definiciones


def agregar_features_candidatas(df: pd.DataFrame) -> pd.DataFrame:
    """Lags/ventanas estrictas por grupo; NaN sin imputar; conserva la base.

    Las ventanas nuevas exigen todos los meses válidos. Las variables BASE
    reutilizadas conservan su semántica preexistente (promedios parciales y
    tendencia cero ante huecos). No se consultan etiquetas futuras.
    """
    requeridas = [*GRUPOS, 'anio', 'mes', 'ocupacion_estimada',
                  'total_pacientes_camas', 'total_camas_disponibles',
                  'total_ingresos', 'total_egresos', 'presion_ingresos_camas',
                  'nivel_riesgo_actual_codificado']
    faltantes = set(requeridas) - set(df.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas: {sorted(faltantes)}')
    if not df.index.is_unique or df[GRUPOS].isna().any().any():
        raise ValueError('Se requieren índices únicos y grupos no nulos.')
    if set(NUEVAS_FEATURES) & set(df.columns):
        raise ValueError('Las features candidatas ya existen; no se sobrescriben.')
    r = df[requeridas].copy().reset_index(drop=True)
    fechas = pd.to_datetime(dict(year=r.anio, month=r.mes, day=1), errors='raise')
    periodos = fechas.dt.to_period('M')
    if 'periodo_actual' in df:
        if not (pd.PeriodIndex(df.periodo_actual.astype(str), freq='M') == pd.PeriodIndex(periodos)).all():
            raise ValueError('anio/mes no coinciden con periodo_actual.')
    r['_mes'] = periodos.astype('int64')
    if r.duplicated([*GRUPOS, '_mes']).any():
        raise ValueError('Mes duplicado por IPRESS/servicio.')
    r = r.sort_values([*GRUPOS, '_mes'], kind='stable')
    for columna in set(requeridas) - set(GRUPOS) - {'anio', 'mes'}:
        r[columna] = pd.to_numeric(r[columna], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if not r.nivel_riesgo_actual_codificado.dropna().isin([0, 1, 2]).all():
        raise ValueError('El riesgo actual debe ser 0, 1, 2 o ausente.')
    grupos = r.groupby(GRUPOS, sort=False, observed=True)
    # Con meses únicos ordenados, diferencia de k meses en k filas implica
    # que todos los meses intermedios existen; nunca puentea un hueco.
    continuos = {k: r['_mes'].sub(grupos['_mes'].shift(k)).eq(k) for k in range(1, 6)}
    def pasado(columna, k):
        return grupos[columna].shift(k).where(continuos[k])
    def ventana(columna, k):
        return pd.concat([r[columna], *(pasado(columna, j) for j in range(1, k))], axis=1)
    def agregar_ventana(salida, columna, k, operacion):
        valores = ventana(columna, k)
        calculo = valores.std(axis=1, ddof=0) if operacion == 'std' else getattr(valores, operacion)(axis=1)
        r[salida] = calculo.where(valores.notna().all(axis=1))
    for salida, (columna, k) in LAGS.items():
        r[salida] = pasado(columna, k)
    for salida, (columna, lag) in CAMBIOS.items():
        r[salida] = r[columna] - r[lag]
    r['aceleracion_ocupacion'] = r.ocupacion_estimada - 2*r.ocupacion_lag_1m + r.ocupacion_lag_2m
    for salida, columna, k, operacion in [
        ('max_ocupacion_3m', 'ocupacion_estimada', 3, 'max'),
        ('min_ocupacion_3m', 'ocupacion_estimada', 3, 'min'),
        ('desviacion_ocupacion_3m', 'ocupacion_estimada', 3, 'std'),
        ('promedio_ocupacion_6m', 'ocupacion_estimada', 6, 'mean'),
        ('max_presion_3m', 'presion_ingresos_camas', 3, 'max'),
        ('promedio_pacientes_camas_3m', 'total_pacientes_camas', 3, 'mean'),
        ('promedio_camas_disponibles_3m', 'total_camas_disponibles', 3, 'mean')]:
        agregar_ventana(salida, columna, k, operacion)
    for salida, k, medio in [('meses_alto_ultimos_3m', 3, False),
                             ('meses_alto_ultimos_6m', 6, False),
                             ('meses_medio_alto_ultimos_3m', 3, True)]:
        valores = ventana('nivel_riesgo_actual_codificado', k)
        condiciones = valores.ge(1) if medio else valores.eq(2)
        r[salida] = condiciones.sum(axis=1).where(valores.notna().all(axis=1))
    r['meses_consecutivos_alto'] = np.nan
    for _, grupo in grupos:
        racha, anterior = 0, None
        for indice, ordinal, riesgo in zip(grupo.index, grupo['_mes'], grupo.nivel_riesgo_actual_codificado):
            if anterior is None or ordinal != anterior+1:
                racha = 0
            racha = racha+1 if riesgo == 2 else 0
            r.loc[indice, 'meses_consecutivos_alto'] = racha if pd.notna(riesgo) else np.nan
            anterior = ordinal
    r['margen_umbral_alto'] = r.ocupacion_estimada - .85
    r['margen_umbral_medio'] = r.ocupacion_estimada - .70
    r['distancia_absoluta_umbral_alto'] = r.margen_umbral_alto.abs()
    for salida, columna, lag in [('crecimiento_demanda_1m', 'total_pacientes_camas', 'pacientes_camas_lag_1m'),
                                 ('crecimiento_capacidad_1m', 'total_camas_disponibles', 'camas_disponibles_lag_1m')]:
        previo = r[lag].where(r[lag].gt(0) & r[columna].ge(0))
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            r[salida] = ((r[columna]-previo)/previo).replace([np.inf, -np.inf], np.nan)
    r['brecha_crecimiento_demanda_capacidad'] = r.crecimiento_demanda_1m - r.crecimiento_capacidad_1m
    resultado = df.copy()
    resultado[NUEVAS_FEATURES] = r.sort_index()[NUEVAS_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy()
    return resultado
