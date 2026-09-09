import pandas as pd
import pytest
from src import features_binarias as fb
from src import variables_temporales_experimentales as vt


def historia():
    filas=[]
    for codigo,servicio,nivel in [('00000001','MEDICINA',0),('00000002','UCI',2)]:
        for k,periodo in enumerate(pd.period_range('2023-11',periods=8,freq='M')):
            filas.append(dict(codigo_ipress=codigo,servicio_hospitalizacion=servicio,
                anio=periodo.year,mes=periodo.month,periodo_actual=str(periodo),
                ocupacion_estimada=.4+k*.05+nivel*.1,total_pacientes_camas=120+k*15,
                total_camas_disponibles=300+k*10,total_ingresos=30+k,total_egresos=20+k,
                presion_ingresos_camas=3+k*.1,nivel_riesgo_actual_codificado=nivel))
    return pd.DataFrame(filas)


def test_whitelist_62_sin_objetivos_identificadores_o_legacy():
    assert len(fb.COLUMNAS_PREDICTORAS_BINARIAS)==62
    assert len(vt.FEATURES_SIN_LEGACY)==25
    assert len(vt.CONJUNTOS['D'])==32
    assert not set(fb.COLUMNAS_PREDICTORAS_BINARIAS)&set(fb.COLUMNAS_PROHIBIDAS)
    assert fb.validar_predictoras(fb.COLUMNAS_PREDICTORAS_BINARIAS)==fb.COLUMNAS_PREDICTORAS_BINARIAS


@pytest.mark.parametrize('columna',fb.COLUMNAS_PROHIBIDAS)
def test_columnas_prohibidas_rechazadas(columna):
    with pytest.raises(ValueError,match='Whitelist'):
        fb.validar_predictoras([*fb.COLUMNAS_PREDICTORAS_BINARIAS,columna])


def test_modo_binario_sin_legacy_y_default_d_conservado():
    datos=historia(); copia=datos.copy(deep=True)
    legado=vt.agregar_features_candidatas(datos)
    pd.testing.assert_frame_equal(legado,vt.agregar_features_candidatas(datos,incluir_legacy=True))
    nuevo=fb.agregar_features_binarias(datos.drop(columns='nivel_riesgo_actual_codificado'))
    pd.testing.assert_frame_equal(nuevo[vt.FEATURES_SIN_LEGACY],legado[vt.FEATURES_SIN_LEGACY])
    assert not set(fb.DERIVADAS_LEGACY)&set(nuevo)
    pd.testing.assert_frame_equal(datos,copia)


def test_etiquetas_futuras_y_legacy_no_afectan_features():
    datos=historia()
    otro=datos.assign(nivel_riesgo_actual_codificado='invalido',ocupacion_siguiente_mes=999,
        alta_presion_capacidad_siguiente_mes=-1,nivel_riesgo_siguiente_mes_codificado=-999)
    pd.testing.assert_frame_equal(fb.agregar_features_binarias(datos)[vt.FEATURES_SIN_LEGACY],
                                  fb.agregar_features_binarias(otro)[vt.FEATURES_SIN_LEGACY])


def test_datos_futuros_no_afectan_pasado():
    datos=historia(); otro=datos.copy(); futuro=datos.periodo_actual.gt('2024-02')
    otro.loc[futuro,['ocupacion_estimada','total_pacientes_camas','total_camas_disponibles',
                    'total_ingresos','total_egresos','presion_ingresos_camas']]=999
    pd.testing.assert_frame_equal(fb.agregar_features_binarias(datos).loc[~futuro,vt.FEATURES_SIN_LEGACY],
                                  fb.agregar_features_binarias(otro).loc[~futuro,vt.FEATURES_SIN_LEGACY])


def test_grupos_y_huecos():
    datos=historia()
    otro=datos.loc[datos.codigo_ipress.eq('00000001')].assign(servicio_hospitalizacion='OTRO',ocupacion_estimada=.1)
    datos=pd.concat([datos,otro],ignore_index=True)
    quitar=datos.codigo_ipress.eq('00000001')&datos.servicio_hospitalizacion.eq('MEDICINA')&datos.periodo_actual.eq('2024-01')
    datos=datos.loc[~quitar].sample(frac=1,random_state=42)
    r=fb.agregar_features_binarias(datos); febrero=r.loc[r.periodo_actual.eq('2024-02')]
    fila=febrero.loc[febrero.codigo_ipress.eq('00000001')&febrero.servicio_hospitalizacion.eq('MEDICINA')].iloc[0]
    assert pd.isna(fila.ocupacion_lag_1m) and pd.isna(fila.ocupacion_lag_2m) and pd.isna(fila.max_ocupacion_3m)
    assert febrero.loc[febrero.servicio_hospitalizacion.eq('OTRO'),'ocupacion_lag_1m'].iloc[0]==.1
    assert febrero.loc[febrero.codigo_ipress.eq('00000002'),'ocupacion_lag_1m'].iloc[0]==pytest.approx(.7)
    assert r.index.equals(datos.index)


@pytest.mark.parametrize('valor,esperado',[('241800','hospitalizacion'),(' 240100 ','hospitalizacion'),
    ('245600','hospitalizacion'),('250100','cuidados_criticos'),('250200','cuidados_criticos'),
    ('240100|241800','hospitalizacion'),('250100;250200','cuidados_criticos')])
def test_familia(valor,esperado):
    assert fb.derivar_familia_servicio(pd.Series([valor])).iloc[0]==esperado


@pytest.mark.parametrize('valor',['220100','230100','040000','130000','150300','160300','',None,
    '240100|250100','MEDICINA','24abc','240100|'])
def test_familia_invalida(valor):
    with pytest.raises(ValueError): fb.derivar_familia_servicio(pd.Series([valor]))


def test_auditoria_completa():
    t=fb.auditar_features([*fb.COLUMNAS_PREDICTORAS_BINARIAS,fb.TARGET,fb.FUTURO,'otra_siguiente_mes']).set_index('feature')
    assert t.feature_d_actual.sum()==69 and t.permanece.sum()==62
    assert set(t.loc[fb.DERIVADAS_LEGACY,'categoria'])=={'D'}
    assert set(t.loc[[fb.TARGET,fb.FUTURO,'periodo_predicho'],'categoria'])=={'C'}
    assert not t.loc[t.permanece,'usa_futuro'].any()
    assert t.loc['familia_servicio','categoria']=='E'
    assert t.loc['otra_siguiente_mes','categoria']=='C'
    assert 'NO camas libres' in t.loc['ratio_camas_disponibles','origen']
