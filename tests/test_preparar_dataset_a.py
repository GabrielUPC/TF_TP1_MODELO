import json

import numpy as np
import pandas as pd
import pytest

from src import preparar_dataset_a as pa


def raw_a():
    return pd.DataFrame({'CO_IPRESS': ['00000001']*4, 'ANHO': ['2020']*4,
        'MES': ['1', '2', '3', '5'], 'CA_CAMAS': ['10', '20', '30', '50'],
        'CA_MEDICOS_TOTAL': ['20', '30', '40', '60'], 'CA_MEDICOS_RESIDENTES': ['1']*4,
        'CA_ENFERMERAS': ['30', '40', '50', '70'], 'CATEGORIA': ['III-1']*4})


def procesar(raw):
    a, _ = pa.normalizar_a(raw, 'ConsultaA.csv')
    limpio, estados, ambiguas, trazas = pa.tratar_claves(a)
    limpio, calidad = pa.normalizar_numericos(a, limpio)
    return pa.agregar_derivadas(limpio), estados, ambiguas, trazas, calidad


def d_frame():
    return pd.DataFrame({'codigo_ipress': ['1', '00000001', '00000001', '00000001'],
        'anio': ['2020']*4, 'mes': ['1', '1', '2', '4'],
        'periodo_actual': ['2020-01', '2020-01', '2020-02', '2020-04'],
        'servicio_hospitalizacion': ['Medicina', 'Cirugia', 'Medicina', 'Medicina'],
        'total_camas': ['4', '6', '15', '8'], 'otra_columna': ['0.123456789012345678', '', 'x', 'y']})


def test_exactos_una_fila_sin_sumar_con_trazabilidad():
    raw = pd.concat([raw_a(), raw_a().iloc[:1], raw_a().iloc[:1]], ignore_index=True)
    a, estados, ambiguas, trazas, _ = procesar(raw)
    assert len(a) == 4
    assert a.CA_CAMAS.iloc[0] == 10
    assert a.estado_calidad_a.iloc[0] == 'DUPLICADO_EXACTO'
    assert estados.cantidad_eliminada.sum() == 2
    assert estados.filas_originales.iloc[0] == 3
    assert len(trazas) == 6 and ambiguas.empty


@pytest.mark.parametrize('columna,valor', [('CA_CAMAS', '11'), ('CATEGORIA', 'II-1'),
                                         ('CA_OTROS_RECURSOS', '9'), ('CA_CAMAS', '10.0')])
def test_ambiguedad_compara_todas_columnas_antes_de_convertir(columna, valor):
    raw = raw_a()
    if columna not in raw:
        raw[columna] = '0'
    raw = pd.concat([raw, raw.iloc[:1].assign(**{columna: valor})], ignore_index=True)
    a, estados, ambiguas, _, _ = procesar(raw)
    assert a.mes.tolist() == [2, 3, 5]
    assert len(ambiguas) == 2
    assert estados.estado_calidad_a.iloc[0] == 'AMBIGUA'
    assert estados.cantidad_eliminada.sum() == 0


def test_exactos_entre_archivos_no_dependen_procedencia():
    uno, _ = pa.normalizar_a(raw_a(), 'uno.csv')
    dos, _ = pa.normalizar_a(raw_a(), 'dos.csv')
    a, estados, ambiguas, trazas = pa.tratar_claves(pd.concat([uno, dos]))
    assert len(a) == 4 and ambiguas.empty
    assert estados.cantidad_eliminada.sum() == 4
    assert set(trazas._archivo) == {'uno.csv', 'dos.csv'}


@pytest.mark.parametrize('valor,motivo', [('NE_0002', 'NO_NUMERICO_O_NO_FINITO'),
    ('inf', 'NO_NUMERICO_O_NO_FINITO'), ('-2', 'NEGATIVO'), ('', 'NULO')])
def test_invalidos_y_negativos_nan_con_valor_original(valor, motivo):
    raw = raw_a()
    raw.loc[0, 'CA_CAMAS'] = valor
    a, _, _, _, calidad = procesar(raw)
    assert pd.isna(a.CA_CAMAS.iloc[0])
    assert pd.isna(a.medicos_por_cama.iloc[0])
    h = calidad.loc[calidad.variable.eq('CA_CAMAS')].iloc[0]
    assert h.valor_original == valor and h.motivo == motivo


def test_cama_cero_no_infinito_y_no_cambia_cero():
    raw = raw_a()
    raw.loc[0, 'CA_CAMAS'] = '0'
    a, *_ = procesar(raw)
    assert a.CA_CAMAS.iloc[0] == 0
    assert a[list(pa.RATIOS)].iloc[0].isna().all()
    assert not np.isinf(a[pa.CANDIDATAS].to_numpy()).any()


def test_temporales_pasado_exacto_no_huecos_ni_futuro():
    a, *_ = procesar(raw_a())
    assert pd.isna(a.variacion_camas_a_1m.iloc[0])
    assert a.variacion_camas_a_1m.iloc[1:3].tolist() == [10, 10]
    assert pd.isna(a.variacion_camas_a_1m.iloc[3])  # mayo no usa marzo
    modificado = raw_a()
    modificado.loc[2:, ['CA_CAMAS', 'CA_MEDICOS_TOTAL', 'CA_ENFERMERAS']] = '999'
    b, *_ = procesar(modificado)
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])


def test_temporales_no_cruzan_mes_ambiguo_ni_ipress():
    raw = pd.concat([raw_a(), raw_a().iloc[1:2].assign(CA_CAMAS='99'),
                     raw_a().assign(CO_IPRESS='00000002')], ignore_index=True)
    a, *_ = procesar(raw)
    marzo = a.loc[a.codigo_ipress.eq('00000001') & a.mes.eq(3)]
    assert marzo[list(pa.CAMBIOS)].isna().all().all()
    assert a.loc[a.codigo_ipress.eq('00000002')].iloc[0][list(pa.CAMBIOS)].isna().all()


def test_join_conserva_d_ambigua_y_sin_match_nan_no_usa_t_mas_uno():
    raw = pd.concat([raw_a(), raw_a().iloc[:1].assign(CA_CAMAS='99')], ignore_index=True)
    a, estados, *_ = procesar(raw)
    d = d_frame()
    r = pa.unir_d_a(d, a)
    pd.testing.assert_frame_equal(r[d.columns], d)
    assert len(r) == 4
    assert r.tiene_datos_a.tolist() == [0, 0, 1, 0]
    assert r.loc[[0, 1, 3], pa.CANDIDATAS].isna().all().all()
    assert r.CA_CAMAS.iloc[2] == 20
    c = pa.cobertura_analitica(r, estados).iloc[-1]
    assert c.con_a_valida == 1 and c.con_clave_ambigua == 2
    assert c.cobertura_pct == 25


def test_match_valido_no_significa_recursos_completos():
    raw = raw_a()
    raw.loc[0, 'CA_MEDICOS_TOTAL'] = 'NE_0001'
    a, estados, *_ = procesar(raw)
    r = pa.unir_d_a(d_frame(), a)
    assert r.tiene_datos_a.tolist() == [1, 1, 1, 0]
    assert r.CA_MEDICOS_TOTAL.iloc[:2].isna().all()
    assert pa.cobertura_analitica(r, estados).iloc[-1].con_cuatro_recursos_numericos == 1


def test_join_rechaza_a_duplicado_y_d_duplicado():
    a, *_ = procesar(raw_a())
    with pytest.raises(ValueError, match='una sola fila'):
        pa.unir_d_a(d_frame(), pd.concat([a, a.iloc[:1]]))
    with pytest.raises(ValueError, match='no es único'):
        pa.unir_d_a(pd.concat([d_frame(), d_frame().iloc[:1]], ignore_index=True), a)


def test_join_rechaza_periodo_incoherente_y_colision():
    a, *_ = procesar(raw_a())
    with pytest.raises(ValueError, match='periodo_actual'):
        pa.unir_d_a(d_frame().assign(periodo_actual='2021-01'), a)
    with pytest.raises(ValueError, match='ya contiene'):
        pa.unir_d_a(d_frame().assign(CA_CAMAS=999), a)


def test_suma_solo_informe_no_reemplaza_d():
    a, *_ = procesar(raw_a())
    d = d_frame()
    original = d.copy(deep=True)
    tabla, resumen = pa.comparar_camas_detallada(d, a)
    enero = tabla.loc[tabla.mes.eq(1)].iloc[0]
    assert enero.suma_total_camas_d == 10 and enero.max_total_camas_d == 6
    assert enero.CA_CAMAS == 10 and enero.servicios_d == 2
    assert resumen['suma_mas_cercana'] == 1
    pd.testing.assert_frame_equal(d, original)
    d.loc[0, 'total_camas'] = ''
    tabla, _ = pa.comparar_camas_detallada(d, a)
    assert pd.isna(tabla.loc[tabla.mes.eq(1), 'suma_total_camas_d'].iloc[0])


def test_clave_invalida_se_excluye_con_evidencia():
    raw = raw_a()
    raw.loc[0, 'MES'] = '13'
    a, _, _, _, calidad = procesar(raw)
    assert len(a) == 3
    assert 'CLAVE_INVALIDA_EXCLUIDA' in calidad.motivo.tolist()


def test_pipeline_raw_produccion_inmutables_y_artefactos(tmp_path):
    raw = tmp_path/'data/raw/Capacidad'
    raw.mkdir(parents=True)
    fuente = raw/'ConsultaA.csv'
    pd.concat([raw_a(), raw_a().iloc[:1]], ignore_index=True).to_csv(fuente, index=False)
    destino = tmp_path/'data/processed'
    destino.mkdir(parents=True)
    produccion = destino/'dataset_modelo_ipress.csv'
    d_frame().to_csv(produccion, index=False)
    modelo = tmp_path/'models/modelo.joblib'
    modelo.parent.mkdir()
    modelo.write_bytes(b'modelo protegido sin cargar')
    d_raw = raw.parent/'ConsultaD.csv'
    d_raw.write_text('raw D protegido', encoding='utf-8')
    antes = {p: pa.sha256(p) for p in [fuente, produccion, modelo, d_raw]}
    r = pa.preparar(tmp_path)
    assert antes == {p: pa.sha256(p) for p in antes}
    assert r['filas_raw_a'] == 5 and r['filas_analiticas_a'] == 4
    assert r['duplicados_exactos_eliminados'] == 1
    assert r['filas_d_antes'] == r['filas_d_despues'] == 4
    assert r['es_modelo_produccion'] is False and r['sin_multiplicacion'] is True
    calidad = tmp_path/'data/quality/dataset_a'
    for nombre in ['duplicados_exactos_tratados.csv', 'claves_ambiguas_a.csv',
        'calidad_dataset_a_analitico.csv', 'cobertura_dataset_a_analitico.csv',
        'comparacion_camas_a_d_detallada.csv', 'procedencia_dataset_a_analitico.csv']:
        assert (calidad/nombre).exists()
    j = json.loads((calidad/'resumen_preparacion_dataset_a.json').read_text(encoding='utf-8'))
    assert j['raw_inmutables_verificados'] is True
    assert j['variables_candidatas_a'] == pa.CANDIDATAS
    assert (destino/'dataset_a_analitico.csv').exists()
    salida = pd.read_csv(destino/'dataset_modelo_ipress_con_a_experimental.csv', dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(salida[d_frame().columns], d_frame())


def test_sin_raw_a_no_genera_salida(tmp_path):
    with pytest.raises(ValueError, match='No hay archivos'):
        pa.preparar(tmp_path)
    assert not (tmp_path/'data/processed').exists()
