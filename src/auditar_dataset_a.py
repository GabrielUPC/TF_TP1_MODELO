"""Auditoría exploratoria ConsultaA y cruce D LEFT JOIN A; nunca modifica RAW."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from datetime import date
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from .datos_raw import leer_csv, SECTORES_PUBLICOS, _detectar_formato

ROOT = Path(__file__).resolve().parent.parent
CLAVE = ['codigo_ipress', 'anio', 'mes']
VARIABLES = ['CA_CAMAS', 'CA_MEDICOS_TOTAL', 'CA_MEDICOS_RESIDENTES', 'CA_ENFERMERAS']
ALIAS = {
    'codigo_ipress': ['CO_IPRESS', 'CODIGO_IPRESS'], 'anio': ['ANHO', 'ANIO', 'ANO'], 'mes': ['MES'],
    'CA_CAMAS': ['CA_CAMAS'], 'CA_MEDICOS_TOTAL': ['CA_MEDICOS_TOTAL'],
    'CA_MEDICOS_RESIDENTES': ['CA_MEDICOS_RESIDENTES'], 'CA_ENFERMERAS': ['CA_ENFERMERAS'],
    'departamento': ['DEPARTAMENTO', 'DEPARTAMENTO_IPRESS'], 'provincia': ['PROVINCIA', 'PROVINCIA_IPRESS'],
    'distrito': ['DISTRITO', 'DISTRITO_IPRESS'], 'sector': ['SECTOR'],
    'categoria': ['CATEGORIA', 'CATEGORIA_IPRESS'], 'nombre_ipress': ['RAZON_SOC', 'NOMBRE_IPRESS'],
}
COLUMNAS_COBERTURA = ['anio', 'registros_d', 'registros_con_a', 'registros_sin_a', 'cobertura_pct',
    'registros_a_ambiguos', 'registros_a_unicos', 'cobertura_univoca_pct',
    'ipress_d', 'ipress_con_coincidencia', 'ipress_sin_coincidencia']
COLUMNAS_CALIDAD = ['ambito', 'anio', 'variable', 'registros', 'nulos', 'porcentaje_nulo', 'no_numericos',
    'ceros', 'negativos', 'minimo', 'p25', 'mediana', 'media', 'p75', 'maximo']
COLUMNAS_CAMAS = [*CLAVE, 'servicio_hospitalizacion', 'total_camas', 'CA_CAMAS',
    'diferencia', 'diferencia_absoluta', 'diferencia_porcentual', 'estado_match', '_archivo', '_registro']


def texto(valor):
    if pd.isna(valor):
        return ''
    valor = unicodedata.normalize('NFKD', str(valor).strip().upper())
    return ' '.join(''.join(c for c in valor if not unicodedata.combining(c)).split())


def nombre_columna(valor):
    return re.sub(r'[^A-Z0-9]+', '_', texto(valor)).strip('_')


def normalizar_codigo(valor):
    """Formato en memoria: 1-8 dígitos, sufijo decimal .0 permitido; nunca truncar."""
    s = '' if pd.isna(valor) else str(valor).strip()
    if not re.fullmatch(r'[0-9]{1,8}(?:\.0+)?', s):
        return pd.NA
    return s.split('.')[0].zfill(8)


def numero(serie):
    # No interpretar ambiguamente comas de miles/decimales: quedan no numéricas.
    return pd.to_numeric(serie, errors='coerce').replace([np.inf, -np.inf], np.nan)


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def localizar(root):
    archivos_a, archivos_d = [], []
    for directorio, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {'.git', '.venv', '.pytest_cache', '__pycache__', 'quality', 'models'}]
        for nombre in sorted(files):
            p = Path(directorio)/nombre
            if p.suffix.lower() != '.csv':
                continue
            normal = re.sub(r'[^a-z0-9]', '', nombre.lower())
            if normal.startswith('consultaa'):
                archivos_a.append(p)
            elif normal.startswith('consultad'):
                archivos_d.append(p)
            elif 'processed' not in p.parts:
                # También reconocer A por cabecera cuando el archivo fue renombrado.
                with p.open('rb') as f:
                    cabecera = f.readline().decode('utf-8-sig', errors='replace')
                if 'CA_CAMAS' in cabecera.upper() and any(v in cabecera.upper() for v in VARIABLES[1:]):
                    archivos_a.append(p)
    return sorted(archivos_a), sorted(archivos_d)


def leer_a(path):
    encoding, separador = _detectar_formato(Path(path))
    try:
        df = pd.read_csv(path, encoding=encoding, sep=separador, dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        # La muestra inicial puede ser ASCII pero el resto del archivo Latin-1.
        encoding = 'latin1'
        df = pd.read_csv(path, encoding=encoding, sep=separador, dtype=str, keep_default_na=False)
    if df.empty:
        raise ValueError('CSV A sin registros.')
    return df, encoding, separador


def normalizar_a(original, archivo):
    df = original.copy()
    df.columns = [nombre_columna(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('Columnas ambiguas tras normalizar encabezados.')
    equivalencias = {}
    r = pd.DataFrame(index=df.index)
    for canonica, candidatos in ALIAS.items():
        presentes = [c for c in candidatos if c in df]
        if len(presentes) > 1:
            raise ValueError(f'Equivalencia ambigua para {canonica}: {presentes}')
        equivalencias[canonica] = presentes[0] if presentes else None
        r[canonica] = df[presentes[0]] if presentes else pd.Series(pd.NA, index=df.index, dtype='string')
    r['_codigo_original'] = r.codigo_ipress.astype('string')
    r['_anio_original'], r['_mes_original'] = r.anio.copy(), r.mes.copy()
    r['codigo_ipress'] = r.codigo_ipress.map(normalizar_codigo).astype('string')
    for c, minimo, maximo in [('anio', 1900, date.today().year), ('mes', 1, 12)]:
        n = numero(r[c])
        r[c] = n.where(n.between(minimo, maximo) & n.mod(1).eq(0)).astype('Int64')
    r['_clave_valida'] = r[CLAVE].notna().all(axis=1)
    r['_archivo'], r['_registro'] = str(archivo), np.arange(1, len(r)+1)
    # Huella sin procedencia: solo evidencia de igualdad, no deduplicación.
    r['_huella_fila'] = pd.util.hash_pandas_object(df.reindex(sorted(df.columns), axis=1).fillna(''), index=False).astype(str)
    for c in df:
        r['original_' + c] = df[c]
    return r, equivalencias


def calidad_por_anio(a, ambito='todos'):
    filas = []
    for anio, grupo in a.groupby('anio', dropna=False):
        for v in VARIABLES:
            s = grupo[v].astype('string').str.strip()
            nulos = s.isna() | s.str.lower().isin(['', 'null', 'nan', 'na', 'n/a'])
            n = numero(s.mask(nulos))
            filas.append(dict(ambito=ambito, anio=anio, variable=v, registros=len(grupo),
                nulos=int(nulos.sum()), porcentaje_nulo=float(nulos.mean()*100),
                no_numericos=int((n.isna() & ~nulos).sum()), ceros=int(n.eq(0).sum()),
                negativos=int(n.lt(0).sum()), minimo=n.min(), p25=n.quantile(.25), mediana=n.median(),
                media=n.mean(), p75=n.quantile(.75), maximo=n.max()))
    return pd.DataFrame(filas, columns=COLUMNAS_CALIDAD)


def clave_y_duplicados(a):
    validos = a.loc[a._clave_valida].copy()
    tamanios = validos.groupby(CLAVE, dropna=False).size().rename('filas_clave')
    repetidos = tamanios[tamanios.gt(1)]
    duplicados = validos.merge(repetidos, on=CLAVE, how='inner', validate='many_to_one')
    huellas_por_clave = duplicados.groupby(CLAVE)._huella_fila.nunique()
    detalle = []
    identificadores = [c for c in a if c.startswith('original_')]
    variaciones = duplicados.groupby(CLAVE)[identificadores].nunique(dropna=False).gt(1).sum()
    for clave, grupo in islice(duplicados.groupby(CLAVE, dropna=False), 30):
        diferencias = [c.removeprefix('original_') for c in identificadores if grupo[c].fillna('').nunique() > 1]
        detalle.append(dict(zip(CLAVE, clave), filas=len(grupo),
            todas_filas_identicas=grupo._huella_fila.nunique() == 1, columnas_que_difieren=diferencias))
    resumen = {'registros': len(a), 'registros_clave_invalida': int((~a._clave_valida).sum()),
        'claves_unicas': len(tamanios), 'claves_repetidas': len(repetidos),
        'max_filas_por_clave': int(tamanios.max()) if len(tamanios) else 0,
        'claves_repetidas_exactas': int(huellas_por_clave.eq(1).sum()),
        'claves_repetidas_con_diferencias': int(huellas_por_clave.gt(1).sum()),
        'registros_en_claves_repetidas': len(duplicados), 'clave_es_unica': len(repetidos) == 0 and bool(len(validos)) and bool(a._clave_valida.all()),
        'claves_con_variacion_por_columna': {c.removeprefix('original_'): int(n) for c,n in variaciones.items() if n > 0},
        'explicacion': 'Se inspeccionan todas las columnas originales; no se deduplican ni suman claves repetidas.',
        'ejemplos_duplicados': detalle[:30]}
    return validos, tamanios, duplicados, resumen


def preparar_d(path):
    d = leer_csv(Path(path), conservar_originales=True)
    if not set([*CLAVE, 'servicio_hospitalizacion', 'total_camas']).issubset(d.columns):
        raise ValueError('D procesado requiere clave, servicio_hospitalizacion y total_camas.')
    d['_codigo_d_original'] = d.codigo_ipress.copy()
    d['codigo_ipress'] = d.codigo_ipress.map(normalizar_codigo).astype('string')
    for c, minimo, maximo in [('anio', 1900, date.today().year), ('mes', 1, 12)]:
        n = numero(d[c])
        d[c] = n.where(n.between(minimo, maximo) & n.mod(1).eq(0)).astype('Int64')
    if d[CLAVE].isna().any().any():
        raise ValueError('D tiene claves inválidas; no se deben omitir silenciosamente.')
    d['total_camas'] = numero(d.total_camas)
    d['_fila_d'] = np.arange(len(d))
    return d


def cruce_izquierdo(d, a):
    validos, tamanios, _, _ = clave_y_duplicados(a)
    unicos = validos.loc[~validos.duplicated(CLAVE, keep=False)].copy()
    r = d.merge(tamanios.rename('filas_a'), on=CLAVE, how='left', validate='many_to_one', sort=False)
    r['filas_a'] = r.filas_a.fillna(0).astype(int)
    r['estado_match'] = np.select([r.filas_a.eq(0), r.filas_a.gt(1)], ['sin_match', 'ambiguo'], default='unico')
    cols = [*CLAVE, *VARIABLES, '_archivo', '_registro']
    r = r.merge(unicos[cols], on=CLAVE, how='left', validate='many_to_one', sort=False)
    assert len(r) == len(d), 'El LEFT JOIN no debe multiplicar ni eliminar filas D'
    return r.sort_values('_fila_d').reset_index(drop=True)


def cobertura(cruce):
    filas = []
    grupos = [(str(anio), g) for anio, g in cruce.groupby('anio')]
    grupos += [('GLOBAL', cruce)]
    for anio, g in grupos:
        con = g.filas_a.gt(0)
        ipress = set(g.codigo_ipress)
        coincidentes = set(g.loc[con, 'codigo_ipress'])
        filas.append(dict(anio=anio, registros_d=len(g), registros_con_a=int(con.sum()),
            registros_sin_a=int((~con).sum()), cobertura_pct=float(con.mean()*100),
            registros_a_ambiguos=int(g.filas_a.gt(1).sum()), registros_a_unicos=int(g.filas_a.eq(1).sum()),
            cobertura_univoca_pct=float(g.filas_a.eq(1).mean()*100), ipress_d=len(ipress),
            ipress_con_coincidencia=len(coincidentes), ipress_sin_coincidencia=len(ipress-coincidentes)))
    return pd.DataFrame(filas, columns=COLUMNAS_COBERTURA)


def comparar_camas(cruce):
    r = cruce[COLUMNAS_CAMAS[:6] + ['estado_match', '_archivo', '_registro']].copy()
    r['CA_CAMAS'] = numero(r.CA_CAMAS)
    valido = r.estado_match.eq('unico') & r.total_camas.ge(0) & r.CA_CAMAS.ge(0)
    r['diferencia'] = (r.total_camas-r.CA_CAMAS).where(valido)
    r['diferencia_absoluta'] = r.diferencia.abs()
    denominador = r.CA_CAMAS.where(valido & r.CA_CAMAS.gt(0))
    r['diferencia_porcentual'] = r.diferencia.div(denominador)*100
    pares = r.loc[valido]
    porcentajes = r.diferencia_porcentual.dropna()
    variacion_servicios = cruce.groupby(CLAVE).total_camas.nunique()
    correlacion = pares.total_camas.corr(pares.CA_CAMAS) if len(pares) > 1 and pares.total_camas.nunique() > 1 and pares.CA_CAMAS.nunique() > 1 else None
    resumen = {'interpretacion': 'Comparación descriptiva de fila/servicio D con total declarado A; NO equivalencia semántica ni suma de servicios.',
        'pares_validos': len(pares), 'pares_porcentaje_A_positivo': len(porcentajes),
        'coincidencia_exacta_pct': float(pares.diferencia.eq(0).mean()*100) if len(pares) else None,
        'dentro_5_pct': float(porcentajes.abs().le(5).mean()*100) if len(porcentajes) else None,
        'dentro_10_pct': float(porcentajes.abs().le(10).mean()*100) if len(porcentajes) else None,
        'correlacion_descriptiva_pearson': correlacion,
        'claves_d_con_camas_distintas_entre_servicios': int(variacion_servicios.gt(1).sum()),
        'evidencia_semantica': 'D conserva servicio; variación entre servicios sugiere granularidad distinta. Confirmar diccionario antes de equiparar CA_CAMAS.',
        'ejemplos_discrepancias': r.loc[valido].sort_values('diferencia_absoluta', ascending=False).head(20).to_dict(orient='records')}
    return r[COLUMNAS_CAMAS], resumen


def candidatas_y_cambios(a):
    validos, tamanios, _, _ = clave_y_duplicados(a)
    u = validos.loc[~validos.duplicated(CLAVE, keep=False)].copy()
    for v in VARIABLES:
        u[v] = numero(u[v])
    formulas = {}
    for salida, numerador in [('medicos_por_cama', 'CA_MEDICOS_TOTAL'), ('enfermeras_por_cama', 'CA_ENFERMERAS'), ('residentes_por_cama', 'CA_MEDICOS_RESIDENTES')]:
        u[salida] = u[numerador].where(u[numerador].ge(0))/u.CA_CAMAS.where(u.CA_CAMAS.gt(0))
        formulas[salida] = {'formula': f'{numerador}/CA_CAMAS; NaN si camas<=0 o numerador inválido/negativo', 'filas_calculables': int(u[salida].notna().sum())}
    u['_ordinal'] = u.anio*12+u.mes
    u = u.sort_values(['codigo_ipress', '_ordinal'])
    grupos = u.groupby('codigo_ipress')
    continuo = u._ordinal.sub(grupos._ordinal.shift(1)).eq(1)
    cambios = []
    for v in VARIABLES:
        anterior = grupos[v].shift(1).where(continuo)
        diferencia = u[v]-anterior
        valido = u[v].ge(0) & anterior.ge(0)
        relativos = diferencia.abs()/anterior.where(anterior.gt(0))
        alarma = valido & diferencia.abs().ge(10) & (relativos.ge(1) | anterior.eq(0))
        for i in u.index[alarma]:
            cambios.append({**u.loc[i, [*CLAVE, '_archivo', '_registro']].to_dict(), 'variable': v,
                'anterior': anterior.loc[i], 'actual': u.loc[i, v], 'diferencia': diferencia.loc[i]})
        formulas['variacion_' + v + '_1m'] = {'formula': f'{v}(t)-{v}(t-1), misma IPRESS, meses exactos y claves únicas',
            'filas_calculables': int((valido & diferencia.notna()).sum())}
    return u, pd.DataFrame(cambios, columns=[*CLAVE, '_archivo', '_registro', 'variable', 'anterior', 'actual', 'diferencia']), formulas


def guardar_json(path, datos):
    limpio = json.loads(pd.Series({'reporte': datos}).to_json(force_ascii=False))['reporte']
    Path(path).write_text(json.dumps(limpio, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def auditar(root=ROOT):
    root = Path(root).resolve()
    archivos_a, archivos_d = localizar(root)
    d_path = root/'data/processed/dataset_modelo_ipress.csv'
    fuentes = [*archivos_a, *archivos_d, *([d_path] if d_path.exists() else [])]
    hashes = {str(p.relative_to(root)): sha256(p) for p in fuentes}
    destino = root/'data/quality/dataset_a'
    destino.mkdir(parents=True, exist_ok=True)
    inventario, tablas, errores = [], [], []
    for path in archivos_a:
        print(f'Auditoría A: {path.name}', flush=True)
        try:
            original, encoding, separador = leer_a(path)
            a, equivalencias = normalizar_a(original, str(path.relative_to(root)))
            tablas.append(a)
            inventario.append({'archivo': str(path.relative_to(root)), 'filas': len(a), 'encoding': encoding, 'separador': separador,
                'anios': json.dumps(sorted(a.anio.dropna().astype(int).unique().tolist())),
                'columnas': json.dumps(original.columns.tolist(), ensure_ascii=False),
                'equivalencias': json.dumps(equivalencias, ensure_ascii=False),
                'otras_columnas_capacidad_rrhh': json.dumps([c for c in original if nombre_columna(c).startswith('CA_') and nombre_columna(c) not in VARIABLES]),
                'claves_invalidas': int((~a._clave_valida).sum()), 'sha256': hashes[str(path.relative_to(root))], 'error': ''})
        except (ValueError, OSError) as e:
            errores.append({'archivo': str(path.relative_to(root)), 'error': str(e)})
            inventario.append({'archivo': str(path.relative_to(root)), 'error': str(e)})
    pd.DataFrame(inventario, columns=['archivo', 'filas', 'encoding', 'separador', 'anios', 'columnas', 'equivalencias', 'otras_columnas_capacidad_rrhh', 'claves_invalidas', 'sha256', 'error']).to_csv(destino/'resumen_archivos_a.csv', index=False)
    reporte = {'fase': 'exploratoria_sin_entrenamiento', 'archivos_a_encontrados': [str(p.relative_to(root)) for p in archivos_a],
        'archivos_d_encontrados': [str(p.relative_to(root)) for p in archivos_d], 'fuente_cruce_d': str(d_path.relative_to(root)),
        'denominador_d': 'D procesado actual, por servicio y mes; ya filtrado y tratado, con pareja t+1. NO es todo el RAW nacional.',
        'fuentes_sha256': hashes, 'errores_lectura': errores, 'anios_disponibles': [], 'filas_totales_a': None,
        'granularidad': None, 'cobertura_por_anio': None, 'cobertura_global': None, 'calidad_variables': None,
        'comparacion_camas': None, 'variables_candidatas': None, 'apto_para_experimento_modelo': False,
        'temporalidad': 'Cruzar A(t) con D(t) o historial anterior; nunca A(t+1). La disponibilidad al cierre debe confirmarse con fecha de publicación.',
        'problemas': [], 'razon': ''}
    calidad = pd.DataFrame(columns=COLUMNAS_CALIDAD)
    duplicados = pd.DataFrame(columns=[*CLAVE, '_archivo', '_registro', 'filas_clave'])
    cobertura_df = pd.DataFrame(columns=COLUMNAS_COBERTURA)
    sin_match = pd.DataFrame(columns=['sentido', 'anio', 'codigo_ipress', 'mes', 'motivo'])
    camas = pd.DataFrame(columns=COLUMNAS_CAMAS)
    cambios = pd.DataFrame(columns=[*CLAVE, '_archivo', '_registro', 'variable', 'anterior', 'actual', 'diferencia'])
    if not tablas:
        reporte.update(estado='bloqueado_sin_datos_a', razon='No se encontraron archivos ConsultaA legibles dentro del repositorio. No se interpreta la ausencia como cobertura cero.',
                       problemas=['ConsultaA ausente o ilegible; se requiere su ubicación dentro del alcance autorizado.'])
    else:
        a = pd.concat(tablas, ignore_index=True)
        print('Analizando granularidad y calidad A...', flush=True)
        validos, tamanios, duplicados, granularidad = clave_y_duplicados(a)
        calidad = calidad_por_anio(a)
        acumulada = calidad.groupby('variable')[['registros', 'nulos', 'ceros', 'negativos', 'no_numericos']].sum()
        acumulada['porcentaje_nulo'] = acumulada.nulos/acumulada.registros*100
        acumulada['porcentaje_no_numerico'] = acumulada.no_numericos/acumulada.registros*100
        reporte['calidad_global'] = acumulada.to_dict(orient='index')
        reporte['ejemplos_valores_no_numericos'] = {v: a.loc[numero(a[v]).isna() & a[v].notna(), v].value_counts().head(10).to_dict() for v in VARIABLES}
        reporte['meses_por_anio'] = {str(anio): sorted(g.mes.dropna().astype(int).unique().tolist()) for anio,g in a.groupby('anio')}
        _, cambios, candidatas = candidatas_y_cambios(a)
        reporte.update(estado='auditoria_preliminar', filas_totales_a=len(a),
            anios_disponibles=sorted(a.anio.dropna().astype(int).unique().tolist()), granularidad=granularidad,
            calidad_variables=calidad.to_dict(orient='records'), variables_candidatas=candidatas,
            cambios_esquema=len({f.get('columnas') for f in inventario if not f.get('error')}) > 1,
            criterio_salto_mensual='Alerta exploratoria: cambio absoluto >=10 y relativo >=100%, o previo=0; solo t-1 contiguo y claves únicas. No es regla de exclusión.',
            codigos_normalizados=int(a._codigo_original.str.strip().ne(a.codigo_ipress).fillna(False).sum()),
            ejemplos_normalizacion=a.loc[a._codigo_original.str.strip().ne(a.codigo_ipress).fillna(False), ['_codigo_original', 'codigo_ipress', '_archivo', '_registro']].head(20).to_dict(orient='records'))
        if d_path.exists():
            print('Cruzando D LEFT JOIN A, sin sumar claves ambiguas...', flush=True)
            d = preparar_d(d_path)
            cruce = cruce_izquierdo(d, a)
            cobertura_df = cobertura(cruce)
            camas, reporte['comparacion_camas'] = comparar_camas(cruce)
            reporte['cobertura_por_anio'] = cobertura_df.loc[cobertura_df.anio.ne('GLOBAL')].to_dict(orient='records')
            reporte['cobertura_global'] = cobertura_df.loc[cobertura_df.anio.eq('GLOBAL')].iloc[0].to_dict()
            # Filtro respaldado por el alcance existente de D, sin inventar categoría hospitalaria.
            vista = a.loc[a.departamento.map(texto).eq('LIMA') & a.provincia.map(texto).eq('LIMA')
                & a.sector.map(texto).isin({texto(s) for s in SECTORES_PUBLICOS})
                & a.codigo_ipress.isin(d.codigo_ipress)].copy()
            reporte['filtro_vista_analitica'] = {'regla': 'Departamento=LIMA, provincia=LIMA, sector público según datos_raw.SECTORES_PUBLICOS, código con hospitalización observada en D.',
                'justificacion': 'Mismo ámbito geográfico/público del proyecto. Hospitalaria por presencia en D 24/25, no por inferir categorías desconocidas.',
                'filas': len(vista), 'categorias_observadas': a.categoria.fillna('(ausente)').value_counts().to_dict(),
                'sectores_observados': a.sector.fillna('(ausente)').value_counts().to_dict(),
                'departamentos_observados': a.departamento.fillna('(ausente)').value_counts().to_dict(),
                'provincias_observadas': a.provincia.fillna('(ausente)').value_counts().to_dict(),
                'limitacion': 'No identifica hospitales sin actividad presente en el D procesado; campos ausentes no se imputan.'}
            vista[[*CLAVE, 'departamento', 'provincia', 'distrito', 'sector', 'categoria', *VARIABLES, '_archivo', '_registro']].to_csv(destino/'vista_analitica_a.csv', index=False)
            calidad = pd.concat([calidad, calidad_por_anio(vista, 'vista_lima_publica_hospitalaria')], ignore_index=True)
            claves_vista = pd.MultiIndex.from_frame(vista.loc[vista._clave_valida, CLAVE])
            presentes_vista = pd.MultiIndex.from_frame(cruce[CLAVE]).isin(claves_vista)
            reporte['filas_d_con_a_fuera_vista_analitica'] = int((cruce.filas_a.gt(0) & ~presentes_vista).sum())
            faltas = []
            for anio, grupo_d in d.groupby('anio'):
                grupo_a = validos.loc[validos.anio.eq(anio)]
                for c in sorted(set(grupo_d.codigo_ipress)-set(grupo_a.codigo_ipress)):
                    faltas.append(dict(sentido='D_sin_A', anio=anio, codigo_ipress=c, mes=None, motivo='Código sin A ese año'))
                for c in sorted(set(grupo_a.codigo_ipress)-set(grupo_d.codigo_ipress)):
                    faltas.append(dict(sentido='A_sin_D', anio=anio, codigo_ipress=c, mes=None, motivo='Código sin D ese año (A nacional vs D filtrado)'))
            for _, fila in cruce.loc[cruce.filas_a.eq(0), CLAVE].drop_duplicates().iterrows():
                faltas.append(dict(sentido='D_mes_sin_A', **fila.to_dict(), motivo='Sin coincidencia exacta del mes'))
            sin_match = pd.DataFrame(faltas, columns=sin_match.columns)
            reporte['anios_d_sin_a'] = sorted(set(d.anio.astype(int))-set(validos.anio.astype(int)))
            reporte['anios_a_sin_d'] = sorted(set(validos.anio.astype(int))-set(d.anio.astype(int)))
            reporte['ipress_d_sin_a_global'] = sorted(set(d.codigo_ipress)-set(validos.codigo_ipress))
            reporte['ipress_a_sin_d_global'] = sorted(set(validos.codigo_ipress)-set(d.codigo_ipress))
            reporte['codigos_d_reformateados'] = int(d._codigo_d_original.ne(d.codigo_ipress).sum())
            # Mismo nombre con códigos distintos es solo una pista, nunca una equivalencia.
            nombres = pd.concat([validos[['codigo_ipress', 'nombre_ipress']], d[['codigo_ipress', 'nombre_ipress']]]) if 'nombre_ipress' in d else validos[['codigo_ipress', 'nombre_ipress']]
            nombres = nombres.assign(nombre_normalizado=nombres.nombre_ipress.map(texto))
            pistas = nombres.loc[nombres.nombre_normalizado.ne('')].groupby('nombre_normalizado').codigo_ipress.agg(lambda s: sorted(set(s)))
            reporte['posibles_cambios_codigo_sin_resolver'] = pistas[pistas.map(len).gt(1)].head(30).to_dict()
        else:
            reporte['problemas'].append('No existe D procesado para el cruce solicitado.')
        reporte['problemas'] += ['No sumar claves A repetidas: decidir granularidad con evidencia de columnas.',
            'La comparación de camas es descriptiva por servicio; falta confirmar diccionario/unidades y alcance de CA_CAMAS.',
            'No se conocen fechas de disponibilidad/publicación; no se garantiza acceso a A(t) al cierre de t.',
            'Preparación actual busca D directamente en data/raw; D ahora en subcarpeta. Ajuste pendiente fuera de esta auditoría.']
        reporte['razon'] = 'A muestra potencial como covariables institucionales, pero no está listo para uso automático: acordar tratamiento de coincidencias ambiguas/no numéricas, no equiparar camas A con camas del servicio D y confirmar disponibilidad de A(t). Esta auditoría no demuestra mejora predictiva.'
    for nombre, tabla in [('calidad_variables_a.csv', calidad), ('duplicados_clave_a.csv', duplicados),
                           ('cobertura_cruce_a_d.csv', cobertura_df), ('ipress_sin_match_a_d.csv', sin_match),
                           ('comparacion_camas_a_d.csv', camas), ('cambios_mensuales_a.csv', cambios)]:
        tabla.to_csv(destino/nombre, index=False)
    if hashes != {str(p.relative_to(root)): sha256(p) for p in fuentes}:
        raise RuntimeError('Un archivo de origen cambió durante la auditoría.')
    reporte['raw_inmutables_verificados'] = True
    guardar_json(destino/'resumen_auditoria_a.json', reporte)
    return reporte


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    r = auditar()
    print(json.dumps({k: r[k] for k in ['estado', 'anios_disponibles', 'filas_totales_a', 'granularidad',
         'cobertura_global', 'apto_para_experimento_modelo', 'razon']}, ensure_ascii=False, default=str, indent=2))


if __name__ == '__main__':
    main()
