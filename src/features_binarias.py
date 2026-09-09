"""Whitelist binaria auditada: 62 features; ninguna etiqueta o margen legacy."""
from __future__ import annotations

import re
import pandas as pd

if __package__:
    from .variables_temporales import (
        COLUMNAS_TEMPORALES, COLUMNAS_PROMEDIO_MOVIL, COLUMNAS_TENDENCIA,
        agregar_variables_temporales, preparar_registro_con_historial,
    )
    from .variables_temporales_experimentales import (
        FEATURES_SIN_LEGACY, RIESGO, MARGENES, CONJUNTOS, LAGS, CAMBIOS,
        CRECIMIENTO, agregar_features_candidatas, definiciones_features,
    )
else:
    from variables_temporales import (
        COLUMNAS_TEMPORALES, COLUMNAS_PROMEDIO_MOVIL, COLUMNAS_TENDENCIA,
        agregar_variables_temporales, preparar_registro_con_historial,
    )
    from variables_temporales_experimentales import (
        FEATURES_SIN_LEGACY, RIESGO, MARGENES, CONJUNTOS, LAGS, CAMBIOS,
        CRECIMIENTO, agregar_features_candidatas, definiciones_features,
    )

TARGET = 'alta_presion_capacidad_siguiente_mes'
FUTURO = 'ocupacion_siguiente_mes'
# Contrato BASE explícito: no importar el motor multiclase y sus ponderaciones.
COLUMNAS_BASE = [
    'anio', 'mes', 'departamento', 'provincia', 'distrito', 'sector',
    'categoria_ipress', 'servicio_hospitalizacion', 'total_ingresos',
    'total_egresos', 'total_estancias', 'total_pacientes_camas', 'total_camas',
    'total_camas_disponibles', 'total_fallecidos', 'dias_mes', 'promedio_estancia',
    'tasa_fallecidos', 'ratio_camas_disponibles', 'ocupacion_estimada',
    'presion_ingresos_camas', 'rotacion_camas', 'diferencia_ingresos_egresos',
]
FEATURES_D_BINARIAS = FEATURES_SIN_LEGACY
COLUMNAS_PREDICTORAS_BINARIAS = [*COLUMNAS_BASE, *COLUMNAS_TEMPORALES, *FEATURES_D_BINARIAS]
ETIQUETAS_LEGACY = ['nivel_riesgo', 'nivel_riesgo_codificado', 'nivel_riesgo_actual',
    'nivel_riesgo_actual_codificado', 'nivel_riesgo_siguiente_mes', 'nivel_riesgo_siguiente_mes_codificado']
DERIVADAS_LEGACY = [*RIESGO, *MARGENES]
SOLO_TRAZABILIDAD = ['codigo_ipress', 'id_hospitalizacion', 'ubigeo', 'nombre_ipress',
    'archivo_origen', 'periodo_actual', 'familia_servicio']
COLUMNAS_PROHIBIDAS = [TARGET, FUTURO, 'periodo_predicho', *ETIQUETAS_LEGACY,
    *DERIVADAS_LEGACY, *SOLO_TRAZABILIDAD]


def agregar_features_binarias(df):
    """Calcular sobre meses válidos antes del target; sin leer etiquetas legacy."""
    return agregar_features_candidatas(df, incluir_legacy=False)


def preparar_entrada_binaria(actual, historial):
    """Construye las 62 features con información disponible hasta el mes t.

    Reutiliza las fórmulas temporales aprobadas. ``historial_completo`` solo es
    verdadero cuando existen los cinco meses calendario exactos anteriores,
    necesarios para la ventana de seis meses (t incluido).
    """
    preparar_registro_con_historial(actual, historial)  # valida grupo, fechas y duplicados
    derivar_familia_servicio(
        pd.Series([registro["id_hospitalizacion"] for registro in [*historial, actual]])
    )  # el modelo solo fue entrenado con familias 24xx/25xx
    base = agregar_variables_temporales(pd.DataFrame([*historial, actual]))
    base = agregar_features_binarias(base)

    periodo_actual = pd.Period(
        f"{int(actual['anio']):04d}-{int(actual['mes']):02d}", freq="M"
    )
    mascara_actual = (
        base["anio"].astype(int).eq(periodo_actual.year)
        & base["mes"].astype(int).eq(periodo_actual.month)
        & base["codigo_ipress"].astype(str).eq(str(actual["codigo_ipress"]))
        & base["servicio_hospitalizacion"].astype(str).eq(
            str(actual["servicio_hospitalizacion"])
        )
    )
    fila = base.loc[mascara_actual].tail(1)
    if len(fila) != 1:
        raise ValueError("No se pudo identificar de forma única el registro actual.")

    periodos_historial = {
        pd.Period(f"{int(registro['anio']):04d}-{int(registro['mes']):02d}", freq="M")
        for registro in historial
    }
    historial_completo = {
        periodo_actual - desplazamiento for desplazamiento in range(1, 6)
    }.issubset(periodos_historial)
    return fila.loc[:, COLUMNAS_PREDICTORAS_BINARIAS].reset_index(drop=True), historial_completo


def validar_predictoras(columnas):
    nombres = list(columnas.columns if isinstance(columnas, pd.DataFrame) else columnas)
    if len(nombres) != len(set(nombres)):
        raise ValueError('Hay predictores duplicados.')
    extras = set(nombres) - set(COLUMNAS_PREDICTORAS_BINARIAS)
    faltantes = set(COLUMNAS_PREDICTORAS_BINARIAS) - set(nombres)
    if extras or faltantes:
        raise ValueError(f'Whitelist binaria inválida; no permitidas={sorted(extras)}, faltantes={sorted(faltantes)}')
    return list(COLUMNAS_PREDICTORAS_BINARIAS)


def derivar_familia_servicio(ids):
    """Solo para estratos; todos los IDs de un grupo deben ser de la misma familia."""
    def familia(valor):
        if pd.isna(valor):
            raise ValueError('ID_HOSPITALIZACION ausente.')
        familias = set()
        for parte in re.split(r'[|;,]', str(valor).strip()):
            match = re.fullmatch(r'((?:24|25)[0-9]*)(?:\.0+)?', parte.strip())
            if not match:
                raise ValueError(f'ID_HOSPITALIZACION fuera de 24/25 o inválido: {valor!r}')
            familias.add(match.group(1)[:2])
        if len(familias) != 1:
            raise ValueError(f'Grupo con familias 24/25 mixtas: {valor!r}')
        return {'24': 'hospitalizacion', '25': 'cuidados_criticos'}[familias.pop()]
    return ids.map(familia).astype('string').rename('familia_servicio')


ORIGEN_RAW = dict(zip(COLUMNAS_BASE[:15], [
    'ANHO', 'MES', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'SECTOR', 'CATEGORIA',
    'HOSPITALIZACION', 'NRO_TOTAL_HOSPIT_ING', 'NRO_TOTAL_HOSPIT_EGR', 'NRO_TOTAL_ESTANCIAS',
    'NRO_TOTAL_PACIENTES_CAMAS (pacientes-día)', 'NRO_TOTAL_CAMAS (camas)',
    'NRO_TOTAL_CAMAS_DISPONIB (días-cama disponibles)', 'NRO_TOTAL_FALLECIDOS']))
FORMULAS = {
    'dias_mes': 'calendario(anio,mes)', 'promedio_estancia': 'total_estancias / total_egresos',
    'tasa_fallecidos': 'total_fallecidos / total_egresos',
    'ratio_camas_disponibles': 'días-cama reportados / (camas * dias_mes); consistencia, NO camas libres',
    'ocupacion_estimada': 'total_pacientes_camas / total_camas_disponibles',
    'presion_ingresos_camas': 'total_ingresos / total_camas', 'rotacion_camas': 'total_egresos / total_camas',
    'diferencia_ingresos_egresos': 'total_ingresos - total_egresos', 'trimestre': '(mes-1)//3+1',
    'semestre': '(mes-1)//6+1', 'mes_sin': 'sin(2*pi*mes/12)', 'mes_cos': 'cos(2*pi*mes/12)',
    'es_fin_de_anio': 'mes == 12'}
REVISAR = {'anio', 'departamento', 'provincia', 'distrito', 'trimestre', 'semestre',
    'es_fin_de_anio', 'dias_mes', 'promedio_estancia', 'tasa_fallecidos', 'ratio_camas_disponibles',
    'presion_ingresos_camas', 'rotacion_camas', 'diferencia_ingresos_egresos',
    *CAMBIOS, 'aceleracion_ocupacion', *CRECIMIENTO}


def auditar_features(columnas_dataset=None):
    """Auditoría semántica de todas las D previas y columnas de dataset; sin estadísticas futuras."""
    disponibles = None if columnas_dataset is None else set(
        columnas_dataset.columns if isinstance(columnas_dataset, pd.DataFrame) else columnas_dataset)
    columnas = list(dict.fromkeys([*COLUMNAS_BASE, *COLUMNAS_TEMPORALES, *CONJUNTOS['D'],
        *COLUMNAS_PROHIBIDAS, *(sorted(disponibles) if disponibles is not None else [])]))
    definiciones, filas = definiciones_features(), []
    for nombre in columnas:
        futuro = nombre in {TARGET, FUTURO, 'periodo_predicho', 'nivel_riesgo_siguiente_mes',
                            'nivel_riesgo_siguiente_mes_codificado'}
        categoria, momento, motivo = 'A', 'cierre de t', 'Disponible al cierre de t; se conserva.'
        origen = ORIGEN_RAW.get(nombre, FORMULAS.get(nombre, ''))
        if nombre in definiciones:
            origen = definiciones[nombre]['formula']
            momento = 't y meses anteriores exactos; ventana completa'
            if nombre in LAGS:
                momento = f't-{LAGS[nombre][1]} meses exactos; intermedios presentes'
        elif nombre in COLUMNAS_PROMEDIO_MOVIL.values():
            fuente = next(k for k,v in COLUMNAS_PROMEDIO_MOVIL.items() if v == nombre)
            origen = f'media({fuente}[t,t-1,t-2]); solo pasado contiguo disponible'
            momento = 't y hasta dos meses previos contiguos; media parcial si falta historia'
        elif nombre in COLUMNAS_TENDENCIA.values():
            fuente = next(k for k,v in COLUMNAS_TENDENCIA.items() if v == nombre)
            origen = f'{fuente}(t)-{fuente}(t-1); 0 técnico si falta t-1 exacto'
            momento = 't y t-1 exacto'
        if nombre in REVISAR:
            categoria, motivo = 'B', 'Válida; revisar redundancia, codificación o estabilidad posteriormente.'
        if nombre in ETIQUETAS_LEGACY or nombre in DERIVADAS_LEGACY:
            categoria, motivo = 'D', 'Fuera: etiqueta multiclase antigua o derivada de cortes 70/85.'
            origen = origen or 'Categorías legacy de ocupación con cortes 0.70/0.85'
            momento = 't+1' if futuro else 't y pasado'
        elif futuro:
            categoria, momento, motivo = 'C', 't+1', 'Prohibida entre predictores del evento futuro.'
            origen = {TARGET: f'{FUTURO} > 0.90', FUTURO: 'ocupacion_estimada del t+1 exacto',
                      'periodo_predicho': 'mes objetivo t+1'}[nombre]
        elif nombre in SOLO_TRAZABILIDAD:
            categoria, motivo = 'E', 'Solo identificación/trazabilidad o estratificación; no predictor.'
            origen = 'ID_HOSPITALIZACION prefijo24/25' if nombre == 'familia_servicio' else 'Identificador o trazabilidad del registro'
        elif nombre not in COLUMNAS_PREDICTORAS_BINARIAS:
            categoria, momento, futuro = 'C', 'sin verificar', None
            origen, motivo = 'Columna no auditada', 'Se excluye hasta verificar origen y tiempo.'
        filas.append(dict(feature=nombre, categoria=categoria, origen=origen,
            momento_disponible=momento, usa_futuro=futuro, permanece=nombre in COLUMNAS_PREDICTORAS_BINARIAS,
            motivo=motivo, feature_d_actual=nombre in [*COLUMNAS_BASE,*COLUMNAS_TEMPORALES,*CONJUNTOS['D']],
            presente_dataset=None if disponibles is None else nombre in disponibles))
    return pd.DataFrame(filas)
