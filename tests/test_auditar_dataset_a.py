import json

import numpy as np
import pandas as pd
import pytest

from src import auditar_dataset_a as aa


def filas_a():
    return pd.DataFrame({'CO_IPRESS': ['00006207', '6207.0', '00006207', '00009999'],
        'ANHO': ['2020']*4, 'MES': ['1', '2', '4', '1'], 'CA_CAMAS': ['10', '0', '12', '20'],
        'CA_MEDICOS_TOTAL': ['20', '-2', 'abc', ''], 'CA_MEDICOS_RESIDENTES': ['2', '3', '4', '5'],
        'CA_ENFERMERAS': ['30', '40', '50', '60'], 'DEPARTAMENTO': ['LIMA']*4,
        'PROVINCIA': ['LIMA']*4, 'DISTRITO': ['LIMA']*4, 'SECTOR': ['MINSA']*4,
        'CATEGORIA': ['III-1']*4, 'RAZON_SOC': ['Hospital']*4})


def d_frame():
    return pd.DataFrame({'codigo_ipress': ['00006207']*5, 'anio': [2020, 2020, 2020, 2020, 2021],
        'mes': [1, 1, 2, 3, 1], 'servicio_hospitalizacion': ['Medicina', 'Cirugia', 'Medicina', 'Medicina', 'Medicina'],
        'total_camas': [5, 8, 10, 10, 10], '_fila_d': range(5)})


@pytest.mark.parametrize('entrada,esperado', [('00006207', '00006207'), ('6207', '00006207'),
    ('6207.0', '00006207'), (' 00006207 ', '00006207')])
def test_codigo_conserva_ceros(entrada, esperado):
    assert aa.normalizar_codigo(entrada) == esperado


@pytest.mark.parametrize('entrada', ['ABC', '123456789', '6207.5', '', None])
def test_codigo_invalido_no_se_trunca(entrada):
    assert pd.isna(aa.normalizar_codigo(entrada))


def test_periodos_validos_y_negativos_no_se_silencian():
    df = filas_a()
    df['MES'] = ['1', '0', '13', '1.5']
    a, _ = aa.normalizar_a(df, 'ConsultaA.csv')
    assert a._clave_valida.tolist() == [True, False, False, False]
    assert a.CA_MEDICOS_TOTAL.iloc[1] == '-2'
    q = aa.calidad_por_anio(a).set_index('variable')
    assert q.loc['CA_MEDICOS_TOTAL', 'negativos'] == 1
    assert q.loc['CA_MEDICOS_TOTAL', 'no_numericos'] == 1
    assert q.loc['CA_MEDICOS_TOTAL', 'nulos'] == 1
    assert q.loc['CA_MEDICOS_TOTAL', 'minimo'] == -2


def test_duplicados_no_se_suman_y_left_join_no_multiplica():
    df = pd.concat([filas_a(), filas_a().iloc[:1].assign(CA_CAMAS='99')], ignore_index=True)
    a, _ = aa.normalizar_a(df, 'ConsultaA.csv')
    original = a.copy(deep=True)
    _, _, duplicados, granularidad = aa.clave_y_duplicados(a)
    assert granularidad['claves_repetidas'] == 1
    assert granularidad['max_filas_por_clave'] == 2
    assert 'CA_CAMAS' in granularidad['ejemplos_duplicados'][0]['columnas_que_difieren']
    r = aa.cruce_izquierdo(d_frame(), a)
    assert len(r) == len(d_frame())
    assert r.estado_match.tolist() == ['ambiguo', 'ambiguo', 'unico', 'sin_match', 'sin_match']
    assert r.loc[:1, 'CA_CAMAS'].isna().all()
    assert set(duplicados.CA_CAMAS) == {'10', '99'}
    pd.testing.assert_frame_equal(a, original)


def test_match_exacto_ipress_anio_mes_nunca_t_mas_uno():
    a, _ = aa.normalizar_a(filas_a(), 'ConsultaA.csv')
    r = aa.cruce_izquierdo(d_frame(), a)
    assert r.CA_CAMAS.iloc[0] == '10'
    assert r.CA_CAMAS.iloc[2] == '0'
    assert pd.isna(r.CA_CAMAS.iloc[3])  # marzo no usa abril
    assert pd.isna(r.CA_CAMAS.iloc[4])  # enero de otro año tampoco usa enero 2020
    cobertura = aa.cobertura(r).set_index('anio')
    assert cobertura.loc['GLOBAL', 'registros_d'] == 5
    assert cobertura.loc['GLOBAL', 'cobertura_pct'] == 60
    assert cobertura.loc['2021', 'ipress_sin_coincidencia'] == 1


def test_ratios_cero_nan_y_variacion_no_cruza_hueco():
    df = filas_a().assign(CA_MEDICOS_TOTAL=['20', '40', '200', ''])
    a, _ = aa.normalizar_a(df, 'ConsultaA.csv')
    u, cambios, formulas = aa.candidatas_y_cambios(a)
    hospital = u.loc[u.codigo_ipress.eq('00006207')]
    assert hospital.medicos_por_cama.iloc[0] == 2
    assert pd.isna(hospital.medicos_por_cama.iloc[1])
    assert not np.isinf(u.medicos_por_cama).any()
    assert cambios.loc[cambios.variable.eq('CA_MEDICOS_TOTAL'), 'mes'].tolist() == [2]
    assert formulas['variacion_CA_CAMAS_1m']['filas_calculables'] == 1


def test_camas_por_servicio_no_se_suman_y_porcentajes_cero_separados():
    a, _ = aa.normalizar_a(filas_a(), 'ConsultaA.csv')
    tabla, resumen = aa.comparar_camas(aa.cruce_izquierdo(d_frame(), a))
    assert tabla.total_camas.iloc[:2].tolist() == [5, 8]
    assert tabla.diferencia_absoluta.iloc[:2].tolist() == [5, 2]
    assert tabla.diferencia_porcentual.iloc[:2].tolist() == [-50, -20]
    assert pd.isna(tabla.diferencia_porcentual.iloc[2])
    assert resumen['claves_d_con_camas_distintas_entre_servicios'] == 1


def test_auditoria_artefactos_raw_inmutables_y_aislada(tmp_path):
    raw = tmp_path/'data/raw/consulta_a'
    raw.mkdir(parents=True)
    a_path = raw/'ConsultaA_2020.csv'
    filas_a().to_csv(a_path, index=False)
    procesado = tmp_path/'data/processed'
    procesado.mkdir(parents=True)
    d_path = procesado/'dataset_modelo_ipress.csv'
    d_frame().drop(columns='_fila_d').to_csv(d_path, index=False)
    antes = {p: aa.sha256(p) for p in (a_path, d_path)}
    r = aa.auditar(tmp_path)
    assert r['filas_totales_a'] == 4
    assert r['anios_disponibles'] == [2020]
    assert r['cobertura_global']['cobertura_pct'] == 60
    assert r['raw_inmutables_verificados'] is True
    assert antes == {p: aa.sha256(p) for p in antes}
    salida = tmp_path/'data/quality/dataset_a'
    for nombre in ['resumen_archivos_a.csv', 'calidad_variables_a.csv', 'duplicados_clave_a.csv',
        'cobertura_cruce_a_d.csv', 'ipress_sin_match_a_d.csv', 'comparacion_camas_a_d.csv', 'resumen_auditoria_a.json']:
        assert (salida/nombre).exists()
    j = json.loads((salida/'resumen_auditoria_a.json').read_text(encoding='utf-8'))
    assert j['apto_para_experimento_modelo'] is False


def test_sin_a_no_inventa_cobertura_cero(tmp_path):
    r = aa.auditar(tmp_path)
    assert r['estado'] == 'bloqueado_sin_datos_a'
    assert r['filas_totales_a'] is None
    assert r['cobertura_global'] is None
    assert r['apto_para_experimento_modelo'] is False


def test_latin1_despues_de_muestra_ascii_se_lee_sin_tocar_fuente(tmp_path):
    path = tmp_path/'ConsultaA_2015.csv'
    df = pd.concat([filas_a().iloc[:1]]*100, ignore_index=True)
    df.loc[99, 'RAZON_SOC'] = 'HOSPITAL NIÑO'
    path.write_bytes(df.to_csv(index=False).encode('latin1'))
    antes = aa.sha256(path)
    leido, encoding, _ = aa.leer_a(path)
    assert len(leido) == 100
    assert leido.iloc[-1].RAZON_SOC == 'HOSPITAL NIÑO'
    assert encoding == 'latin1'
    assert aa.sha256(path) == antes
