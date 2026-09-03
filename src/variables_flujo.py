"""Diecisiete candidatas de flujo/permanencia, aisladas del dataset de producción."""
import numpy as np
import pandas as pd

GRUPOS = ['codigo_ipress', 'servicio_hospitalizacion']
BLOQUES = {
    'A': ['balance_flujo_mes', 'balance_flujo_acumulado_3m', 'promedio_balance_flujo_3m',
          'meses_ingresos_mayor_egresos_3m', 'ratio_egresos_ingresos'],
    'B': ['crecimiento_ingresos_1m', 'crecimiento_ingresos_2m',
          'crecimiento_pacientes_cama_1m', 'crecimiento_pacientes_cama_2m'],
    'C': ['estancia_promedio_actual', 'estancia_promedio_lag_1m',
          'cambio_estancia_promedio_1m', 'promedio_estancia_3m'],
    'D': ['aceleracion_ingresos', 'aceleracion_pacientes_cama'],
    'E': ['racha_ocupacion_creciente_3m', 'meses_ocupacion_sobre_80_ultimos_3m'],
}
FEATURES = [c for bloque in BLOQUES.values() for c in bloque]
VARIANTES = {'D': [], 'D+FLUJO': BLOQUES['A'],
    'D+DEMANDA': [*BLOQUES['A'], *BLOQUES['B']],
    'D+PERMANENCIA': [*BLOQUES['A'], *BLOQUES['B'], *BLOQUES['C']],
    'D+DINAMICA': FEATURES}
FUENTES = ['total_ingresos', 'total_egresos', 'total_estancias', 'total_pacientes_camas', 'ocupacion_estimada']


def definiciones():
    formulas = {
        'balance_flujo_mes': 'total_ingresos(t) - total_egresos(t)',
        'balance_flujo_acumulado_3m': 'suma balance_flujo_mes(t,t-1,t-2)',
        'promedio_balance_flujo_3m': 'media balance_flujo_mes(t,t-1,t-2)',
        'meses_ingresos_mayor_egresos_3m': 'suma de indicadores total_ingresos > total_egresos en t,t-1,t-2',
        'ratio_egresos_ingresos': 'total_egresos(t) / total_ingresos(t)',
        'estancia_promedio_actual': 'total_estancias(t) / total_egresos(t)',
        'estancia_promedio_lag_1m': 'estancia_promedio_actual(t-1)',
        'cambio_estancia_promedio_1m': 'estancia_promedio_actual(t) - estancia_promedio_actual(t-1)',
        'promedio_estancia_3m': 'media estancia_promedio_actual(t,t-1,t-2)',
        'aceleracion_ingresos': 'total_ingresos(t) - 2*total_ingresos(t-1) + total_ingresos(t-2)',
        'aceleracion_pacientes_cama': 'total_pacientes_camas(t) - 2*total_pacientes_camas(t-1) + total_pacientes_camas(t-2)',
        'racha_ocupacion_creciente_3m': '0 si ocupacion(t)<=ocupacion(t-1); si aumenta: 1 + indicador ocupacion(t-1)>ocupacion(t-2); exige tres meses válidos',
        'meses_ocupacion_sobre_80_ultimos_3m': 'suma de indicadores ocupacion_estimada > 0.80 en t,t-1,t-2 (estricto)',
    }
    for nombre, fuente in [('ingresos', 'total_ingresos'), ('pacientes_cama', 'total_pacientes_camas')]:
        for k in (1, 2):
            formulas[f'crecimiento_{nombre}_{k}m'] = f'({fuente}(t) - {fuente}(t-{k})) / {fuente}(t-{k})'
    return {c: {'formula': formulas[c], 'grupo': GRUPOS,
                'bloque': next(b for b, cols in BLOQUES.items() if c in cols),
                'ausentes': 'NaN si falta operando/mes requerido, denominador<=0 o resultado no finito; sin imputar',
                'continuidad': 'Meses calendario exactos; t y pasado únicamente'} for c in FEATURES}


def agregar_features_flujo(df):
    requeridas = [*GRUPOS, 'anio', 'mes', *FUENTES]
    if set(requeridas)-set(df):
        raise ValueError(f'Faltan columnas: {sorted(set(requeridas)-set(df))}')
    if not df.index.is_unique or df[GRUPOS].isna().any().any():
        raise ValueError('Índices únicos y grupos no nulos requeridos.')
    if set(FEATURES) & set(df):
        raise ValueError('Las features de flujo ya existen; no sobrescribir.')
    r = df[requeridas].copy().reset_index(drop=True)
    fechas = pd.to_datetime(dict(year=r.anio, month=r.mes, day=1), errors='raise')
    periodos = fechas.dt.to_period('M')
    if periodos.isna().any():
        raise ValueError('Mes inválido.')
    if 'periodo_actual' in df and not (pd.PeriodIndex(df.periodo_actual.astype(str), freq='M') == pd.PeriodIndex(periodos)).all():
        raise ValueError('anio/mes no coincide con periodo_actual.')
    r['_mes'] = periodos.astype('int64')
    if r.duplicated([*GRUPOS, '_mes']).any():
        raise ValueError('Mes duplicado por IPRESS y servicio.')
    r = r.sort_values([*GRUPOS, '_mes'], kind='stable')
    for c in FUENTES:
        n = pd.to_numeric(r[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        r[c] = n.where(n.ge(0))  # Solo en la copia de cálculo, sin corregir D.
    def pasado(c, k):
        grupos = r.groupby(GRUPOS, sort=False, observed=True)
        continuo = r._mes.sub(grupos._mes.shift(k)).eq(k)
        return grupos[c].shift(k).where(continuo)
    def ventana(c):
        return pd.concat([r[c], pasado(c, 1), pasado(c, 2)], axis=1)
    def division(numerador, denominador):
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            return (numerador/denominador.where(denominador.gt(0))).replace([np.inf, -np.inf], np.nan)
    r['balance_flujo_mes'] = r.total_ingresos - r.total_egresos
    flujo = ventana('balance_flujo_mes')
    completo = flujo.notna().all(axis=1)
    r['balance_flujo_acumulado_3m'] = flujo.sum(axis=1).where(completo)
    r['promedio_balance_flujo_3m'] = flujo.mean(axis=1).where(completo)
    r['meses_ingresos_mayor_egresos_3m'] = flujo.gt(0).sum(axis=1).where(completo)
    r['ratio_egresos_ingresos'] = division(r.total_egresos, r.total_ingresos)
    for nombre, fuente in [('ingresos', 'total_ingresos'), ('pacientes_cama', 'total_pacientes_camas')]:
        for k in (1, 2):
            previo = pasado(fuente, k)
            r[f'crecimiento_{nombre}_{k}m'] = division(r[fuente]-previo, previo)
    r['estancia_promedio_actual'] = division(r.total_estancias, r.total_egresos)
    r['estancia_promedio_lag_1m'] = pasado('estancia_promedio_actual', 1)
    r['cambio_estancia_promedio_1m'] = r.estancia_promedio_actual-r.estancia_promedio_lag_1m
    estancia = ventana('estancia_promedio_actual')
    r['promedio_estancia_3m'] = estancia.mean(axis=1).where(estancia.notna().all(axis=1))
    for nombre, fuente in [('ingresos', 'total_ingresos'), ('pacientes_cama', 'total_pacientes_camas')]:
        r[f'aceleracion_{nombre}'] = r[fuente]-2*pasado(fuente, 1)+pasado(fuente, 2)
    ocupacion = ventana('ocupacion_estimada')
    completo = ocupacion.notna().all(axis=1)
    crece = ocupacion.iloc[:, 0].gt(ocupacion.iloc[:, 1])
    previo_crece = ocupacion.iloc[:, 1].gt(ocupacion.iloc[:, 2])
    r['racha_ocupacion_creciente_3m'] = (crece.astype(int)*(1+previo_crece.astype(int))).where(completo)
    r['meses_ocupacion_sobre_80_ultimos_3m'] = ocupacion.gt(.80).sum(axis=1).where(completo)
    resultado = df.copy()
    resultado[FEATURES] = r.sort_index()[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy()
    return resultado
