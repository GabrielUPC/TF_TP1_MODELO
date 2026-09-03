"""Prepara covariables A y una copia experimental D+A; no entrena ni altera fuentes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .auditar_dataset_a import (
    ROOT, CLAVE, VARIABLES, guardar_json, leer_a, localizar,
    normalizar_a, normalizar_codigo, numero, sha256,
)

VERSION = 'dataset_a_analitico_v1'
RATIOS = {'medicos_por_cama': 'CA_MEDICOS_TOTAL',
          'enfermeras_por_cama': 'CA_ENFERMERAS',
          'residentes_por_cama': 'CA_MEDICOS_RESIDENTES'}
CAMBIOS = {'variacion_camas_a_1m': 'CA_CAMAS',
           'variacion_medicos_1m': 'CA_MEDICOS_TOTAL',
           'variacion_enfermeras_1m': 'CA_ENFERMERAS'}
CANDIDATAS = [*VARIABLES, *RATIOS, *CAMBIOS]
PROCEDENCIA = [*CLAVE, '_archivo', '_registro']


def tratar_claves(a):
    """Igualdad de todas las columnas originales, antes de conversión numérica.

    La procedencia no cuenta como contenido. No se decide igualdad por hashes,
    ni por las cuatro covariables solamente. Una clave ambigua se excluye entera.
    """
    originales = sorted(c for c in a if c.startswith('original_'))
    if not originales:
        raise ValueError('Se necesitan columnas originales para comparar duplicados.')
    validos = a.loc[a._clave_valida].copy()
    conteo = validos.groupby(CLAVE).size().rename('filas_originales')
    versiones = validos.drop_duplicates([*CLAVE, *originales]).groupby(CLAVE).size()
    estados = conteo.to_frame().join(versiones.rename('versiones_distintas')).reset_index()
    estados['estado_calidad_a'] = np.select(
        [estados.versiones_distintas.gt(1), estados.filas_originales.gt(1)],
        ['AMBIGUA', 'DUPLICADO_EXACTO'], default='UNICA')
    estados['cantidad_eliminada'] = np.where(
        estados.estado_calidad_a.eq('DUPLICADO_EXACTO'), estados.filas_originales - 1, 0)
    trazas = validos.merge(estados, on=CLAVE, how='left', validate='many_to_one')
    ambiguas = trazas.loc[trazas.estado_calidad_a.eq('AMBIGUA')].copy()
    # keep=first solo después de comprobar igualdad en TODO el contenido.
    limpias = trazas.loc[trazas.estado_calidad_a.ne('AMBIGUA')].drop_duplicates(CLAVE)
    analitico = limpias[[*CLAVE, *VARIABLES, 'estado_calidad_a']].copy()
    return analitico, estados, ambiguas, trazas


def normalizar_numericos(a, analitico):
    """Hallazgos por registro RAW. Negativos se conservan en evidencia, no como recurso utilizable."""
    hallazgos = []
    for v in VARIABLES:
        n = numero(a[v])
        vacio = a[v].isna() | a[v].astype('string').str.strip().eq('')
        mal = n.isna() | n.lt(0)
        h = a.loc[mal, PROCEDENCIA].copy()
        h['variable'] = v
        h['valor_original'] = a.loc[mal, v]
        h['motivo'] = np.select([vacio.loc[mal], n.loc[mal].lt(0)],
                                ['NULO', 'NEGATIVO'], default='NO_NUMERICO_O_NO_FINITO')
        hallazgos.append(h)
        convertido = numero(analitico[v])
        analitico[v] = convertido.where(convertido.ge(0)).astype(float)
    invalidas = a.loc[~a._clave_valida, PROCEDENCIA].copy()
    invalidas['variable'] = 'clave'
    partes = a.loc[~a._clave_valida, ['_codigo_original', '_anio_original', '_mes_original']].fillna('').astype(str)
    invalidas['valor_original'] = partes._codigo_original + '|' + partes._anio_original + '|' + partes._mes_original
    invalidas['motivo'] = 'CLAVE_INVALIDA_EXCLUIDA'
    hallazgos.append(invalidas)
    return analitico, pd.concat(hallazgos, ignore_index=True)


def agregar_derivadas(a):
    a = a.sort_values(CLAVE).reset_index(drop=True).copy()
    for salida, numerador in RATIOS.items():
        a[salida] = (a[numerador] / a.CA_CAMAS.where(a.CA_CAMAS.gt(0))).replace(
            [np.inf, -np.inf], np.nan)
    ordinal = a.anio * 12 + a.mes
    continuo = ordinal.sub(ordinal.groupby(a.codigo_ipress).shift(1)).eq(1)
    for salida, variable in CAMBIOS.items():
        previo = a.groupby('codigo_ipress')[variable].shift(1)
        a[salida] = a[variable].sub(previo).where(continuo).replace([np.inf, -np.inf], np.nan)
    return a


def claves_d(d):
    requeridas = [*CLAVE, 'servicio_hospitalizacion', 'periodo_actual', 'total_camas']
    if not set(requeridas).issubset(d):
        raise ValueError('D debe contener clave, servicio, periodo_actual y total_camas.')
    k = d[CLAVE].copy()
    k.codigo_ipress = k.codigo_ipress.map(normalizar_codigo).astype('string')
    for c, menor, mayor in [('anio', 1900, 9999), ('mes', 1, 12)]:
        n = numero(k[c])
        k[c] = n.where(n.between(menor, mayor) & n.mod(1).eq(0)).astype('Int64')
    if k.isna().any().any():
        raise ValueError('D tiene claves inválidas.')
    periodo = k.anio.astype(str) + '-' + k.mes.astype(str).str.zfill(2)
    if not periodo.eq(d.periodo_actual.astype(str)).all():
        raise ValueError('periodo_actual de D no coincide con anio/mes.')
    if d.servicio_hospitalizacion.isna().any() or d.servicio_hospitalizacion.astype(str).str.strip().eq('').any():
        raise ValueError('D tiene servicios vacíos.')
    clave_final = k.assign(servicio_hospitalizacion=d.servicio_hospitalizacion,
                          periodo_actual=d.periodo_actual)
    if clave_final.duplicated(['codigo_ipress', 'servicio_hospitalizacion', 'periodo_actual']).any():
        raise ValueError('D no es único por IPRESS, servicio y periodo_actual.')
    return k


def unir_d_a(d, a):
    """LEFT JOIN many_to_one en t exacto; conserva contenido y orden de columnas D."""
    d = d.reset_index(drop=True)
    k = claves_d(d)
    if a.duplicated(CLAVE).any() or a[CLAVE].isna().any().any():
        raise ValueError('A analítico debe tener una sola fila por clave válida.')
    extras = [c for c in a if c not in CLAVE]
    if set([*extras, 'tiene_datos_a']).intersection(d.columns):
        raise ValueError('D ya contiene columnas A; no sobrescribir datos existentes.')
    adjunto = k.merge(a, on=CLAVE, how='left', sort=False, validate='many_to_one', indicator=True)
    if len(adjunto) != len(d):
        raise ValueError('El LEFT JOIN alteró la cantidad de filas D.')
    resultado = d.copy()
    for c in extras:
        resultado[c] = adjunto[c].to_numpy()
    resultado['tiene_datos_a'] = adjunto['_merge'].eq('both').astype(int).to_numpy()
    claves_d(resultado)
    pd.testing.assert_frame_equal(resultado[d.columns], d)
    return resultado


def cobertura_analitica(cruce, estados):
    estado = claves_d(cruce).merge(estados[[*CLAVE, 'estado_calidad_a']],
                                   how='left', on=CLAVE, validate='many_to_one')
    filas = []
    for anio, g in [*cruce.groupby('anio'), ('GLOBAL', cruce)]:
        con = g.tiene_datos_a.eq(1)
        filas.append({'anio': anio, 'filas_d': len(g), 'con_a_valida': int(con.sum()),
            'sin_a_valida': int((~con).sum()), 'cobertura_pct': float(con.mean()*100),
            'con_clave_ambigua': int(estado.loc[g.index, 'estado_calidad_a'].eq('AMBIGUA').sum()),
            'con_cuatro_recursos_numericos': int(g[VARIABLES].notna().all(axis=1).sum())})
    return pd.DataFrame(filas)


def comparar_camas_detallada(d, a):
    """Suma exploratoria SOLO de servicios presentes en D procesado; nunca feature."""
    k = claves_d(d)
    n = numero(d.total_camas)
    k['camas_d'] = n.where(n.ge(0))
    g = k.groupby(CLAVE)
    tabla = g.agg(servicios_d=('camas_d', 'size'), servicios_con_camas_validas=('camas_d', 'count'),
                  suma_total_camas_d=('camas_d', 'sum'), max_total_camas_d=('camas_d', 'max')).reset_index()
    completos = tabla.servicios_d.eq(tabla.servicios_con_camas_validas)
    # Una suma parcial de valores faltantes no debe aparentar un total válido.
    tabla.loc[~completos, ['suma_total_camas_d', 'max_total_camas_d']] = np.nan
    tabla = tabla.merge(a[[*CLAVE, 'CA_CAMAS', 'estado_calidad_a']], on=CLAVE,
                        how='left', validate='one_to_one')
    resumen = {}
    for nombre in ['suma_total_camas_d', 'max_total_camas_d']:
        valido = tabla[nombre].notna() & tabla.CA_CAMAS.notna()
        diferencia = (tabla[nombre] - tabla.CA_CAMAS).where(valido)
        tabla['diferencia_' + nombre] = diferencia
        porcentaje = diferencia / tabla.CA_CAMAS.where(tabla.CA_CAMAS.gt(0)) * 100
        tabla['diferencia_pct_' + nombre] = porcentaje
        pares = tabla.loc[valido]
        resumen[nombre] = {'pares_validos': len(pares), 'pares_a_positivo': int(porcentaje.notna().sum()),
            'coincidencia_exacta_pct': float(diferencia.dropna().eq(0).mean()*100),
            'dentro_10_pct': float(porcentaje.dropna().abs().le(10).mean()*100),
            'mediana_error_absoluto_pct': float(porcentaje.abs().median()),
            'correlacion_pearson': pares[nombre].corr(pares.CA_CAMAS)
                if len(pares) > 1 and pares[nombre].nunique() > 1 and pares.CA_CAMAS.nunique() > 1 else None}
    suma = tabla.diferencia_suma_total_camas_d.abs()
    maximo = tabla.diferencia_max_total_camas_d.abs()
    pares = suma.notna() & maximo.notna()
    resumen.update(grupos_comparables=int(pares.sum()), suma_mas_cercana=int((suma < maximo).sum()),
        maximo_mas_cercano=int((maximo < suma).sum()), empate=int((suma.eq(maximo) & pares).sum()),
        limitacion='Solo servicios presentes en D procesado (filtrado, tratado y con target t+1); '
        'no es inventario completo. Posible solapamiento de servicios. No prueba equivalencia ni reemplaza camas D.')
    return tabla, resumen


def preparar(root=ROOT):
    root = Path(root).resolve()
    raw = root/'data/raw'
    archivos_a, _ = localizar(raw)
    if not archivos_a:
        raise ValueError('No hay archivos ConsultaA en data/raw; no generar dataset vacío.')
    d_path = root/'data/processed/dataset_modelo_ipress.csv'
    protegidos = sorted(raw.rglob('*.csv')) + [d_path] + sorted((root/'models').rglob('*.joblib'))
    hashes = {str(p.relative_to(root)): sha256(p) for p in protegidos}
    tablas = []
    for path in archivos_a:
        print(f'Leyendo A: {path.name}', flush=True)
        original, _, _ = leer_a(path)
        normalizado, equivalencias = normalizar_a(original, str(path.relative_to(root)))
        if any(equivalencias[c] is None for c in [*CLAVE, *VARIABLES]):
            raise ValueError(f'Faltan columnas requeridas en {path.name}.')
        tablas.append(normalizado)
    a = pd.concat(tablas, ignore_index=True)
    print('Clasificando claves y preparando covariables...', flush=True)
    analitico, estados, ambiguas, trazas = tratar_claves(a)
    analitico, hallazgos = normalizar_numericos(a, analitico)
    analitico = agregar_derivadas(analitico)
    # D como texto para conservar valores públicos y precisión sin reconversión.
    d = pd.read_csv(d_path, dtype=str, keep_default_na=False)
    cruce = unir_d_a(d, analitico)
    cobertura = cobertura_analitica(cruce, estados)
    camas, resumen_camas = comparar_camas_detallada(d, analitico)
    exactos = estados.loc[estados.estado_calidad_a.eq('DUPLICADO_EXACTO')]
    procedencia = trazas[[*PROCEDENCIA, 'estado_calidad_a', 'filas_originales', 'cantidad_eliminada']]
    global_ = cobertura.iloc[-1]
    resumen = {'version': VERSION, 'filas_raw_a': len(a), 'filas_analiticas_a': len(analitico),
        'claves_duplicadas_exactas': len(exactos), 'duplicados_exactos_eliminados': int(exactos.cantidad_eliminada.sum()),
        'claves_ambiguas': int(estados.estado_calidad_a.eq('AMBIGUA').sum()),
        'filas_raw_claves_ambiguas': len(ambiguas), 'filas_clave_invalida': int((~a._clave_valida).sum()),
        'registros_d_con_a_valida': int(global_.con_a_valida), 'registros_d_sin_a_valida': int(global_.sin_a_valida),
        'registros_d_clave_ambigua': int(global_.con_clave_ambigua),
        'registros_d_con_cuatro_recursos_numericos': int(global_.con_cuatro_recursos_numericos),
        'cobertura_pct': float(global_.cobertura_pct), 'filas_d_antes': len(d), 'filas_d_despues': len(cruce),
        'sin_multiplicacion': len(d) == len(cruce), 'variables_candidatas_a': CANDIDATAS,
        'es_modelo_produccion': False, 'alcance_a': 'Todas las claves RAW A nacionales no ambiguas; D conserva su alcance existente.',
        'definicion_tiene_datos_a': '1 si hay clave A no ambigua (incluye duplicado exacto resuelto); no garantiza recursos completos.',
        'politica_negativos': 'NaN en analítico; valor original y motivo NEGATIVO en hallazgos. Nunca cero.',
        'igualdad_duplicados': 'Todas las columnas originales como texto, excluyendo procedencia; antes de conversión numérica.',
        'disponibilidad_temporal': 'A(t) solo si estaba disponible al predecir t+1. No A(t+1), imputación futura ni salto de huecos. Fecha de publicación aún no confirmada.',
        'comparacion_camas': resumen_camas, 'fuentes_sha256': hashes,
        'fuentes_a': [str(p.relative_to(root)) for p in archivos_a],
        'fuente_d': str(d_path.relative_to(root))}
    if hashes != {str(p.relative_to(root)): sha256(p) for p in protegidos}:
        raise RuntimeError('Una fuente o modelo cambió durante la preparación; no publicar salidas.')
    destino = root/'data/quality/dataset_a'
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in [('duplicados_exactos_tratados.csv', exactos), ('claves_ambiguas_a.csv', ambiguas),
            ('calidad_dataset_a_analitico.csv', hallazgos), ('cobertura_dataset_a_analitico.csv', cobertura),
            ('comparacion_camas_a_d_detallada.csv', camas), ('procedencia_dataset_a_analitico.csv', procedencia)]:
        tabla.to_csv(destino/nombre, index=False)
    analitico.to_csv(root/'data/processed/dataset_a_analitico.csv', index=False)
    cruce.to_csv(root/'data/processed/dataset_modelo_ipress_con_a_experimental.csv', index=False)
    if hashes != {str(p.relative_to(root)): sha256(p) for p in protegidos}:
        raise RuntimeError('Cambió un archivo protegido durante la escritura.')
    resumen['raw_inmutables_verificados'] = True
    resumen['dataset_d_original_y_modelos_inmutables_verificados'] = True
    guardar_json(destino/'resumen_preparacion_dataset_a.json', resumen)
    return resumen


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    r = preparar()
    print(json.dumps({k: v for k, v in r.items() if k != 'fuentes_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
