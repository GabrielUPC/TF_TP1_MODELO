import json

import pandas as pd
import pytest

from src import preparar_dataset as prep
from src.calidad_datos import auditar_dataframe
from src.datos_raw import (
    RAW_DATA_DIR,
    listar_archivos_csv,
    normalizar_codigo_ipress,
    normalizar_codigos_ipress,
    resumir_normalizacion_codigo_ipress,
)


@pytest.mark.parametrize("entrada,esperado", [
    ("9251", "00009251"),
    ("00009251", "00009251"),
    (" 6207 ", "00006207"),
    ("6207.0", "00006207"),
])
def test_codigo_ipress_canonico_tiene_ocho_caracteres(entrada, esperado):
    assert normalizar_codigo_ipress(entrada) == esperado


@pytest.mark.parametrize("entrada", ["", None, "ABC", "123456789", "6207.5", "-1"])
def test_codigo_ipress_invalido_no_se_trunca_ni_convierte(entrada):
    assert pd.isna(normalizar_codigo_ipress(entrada))


def test_normalizacion_es_string_inmutable_y_trazable():
    original = pd.Series(["9251", "00009251", " 6207 ", "6207.0"], dtype="string")
    copia = original.copy(deep=True)
    resultado = normalizar_codigos_ipress(original)
    assert resultado.tolist() == ["00009251", "00009251", "00006207", "00006207"]
    assert str(resultado.dtype) == "string"
    pd.testing.assert_series_equal(original, copia)
    resumen = resumir_normalizacion_codigo_ipress(original)
    assert resumen["longitud_canonica"] == 8
    assert resumen["filas_transformadas"] == 3
    assert resumen["codigos_distintos_originales"] == 4
    assert resumen["codigos_distintos_canonicos"] == 2


def test_limpieza_unifica_equivalencias_antes_de_agrupar():
    base = {
        "ANHO": ["2019", "2020"], "MES": ["12", "1"],
        "UBIGEO": ["150101"] * 2, "DEPARTAMENTO": ["LIMA"] * 2,
        "PROVINCIA": ["LIMA"] * 2, "DISTRITO": ["LIMA"] * 2,
        "SECTOR": ["MINSA"] * 2, "CATEGORIA": ["III-1"] * 2,
        "CO_IPRESS": ["9251", "00009251"], "RAZON_SOC": ["HOSPITAL"] * 2,
        "ID_HOSPITALIZACION": ["241500"] * 2,
        "HOSPITALIZACION": ["MEDICINA"] * 2,
        "NRO_TOTAL_HOSPIT_ING": ["1"] * 2, "NRO_TOTAL_HOSPIT_EGR": ["1"] * 2,
        "NRO_TOTAL_ESTANCIAS": ["1"] * 2,
        "NRO_TOTAL_PACIENTES_CAMAS": ["1"] * 2,
        "NRO_TOTAL_CAMAS": ["1"] * 2,
        "NRO_TOTAL_CAMAS_DISPONIB": ["31"] * 2,
        "NRO_TOTAL_FALLECIDOS": ["0"] * 2,
    }
    limpio = prep.limpiar_registros(pd.DataFrame(base))
    assert limpio.CO_IPRESS.tolist() == ["00009251", "00009251"]


def test_auditoria_conserva_original_y_publica_codigo_canonico():
    fila = pd.DataFrame({
        "ANHO": ["2019"], "MES": ["1"], "UBIGEO": ["150101"],
        "DEPARTAMENTO": ["LIMA"], "PROVINCIA": ["LIMA"], "DISTRITO": ["LIMA"],
        "SECTOR": ["MINSA"], "CATEGORIA": ["III-1"], "CO_IPRESS": ["9251"],
        "RAZON_SOC": ["HOSPITAL"], "ID_HOSPITALIZACION": ["241500"],
        "HOSPITALIZACION": ["MEDICINA"], "NRO_TOTAL_HOSPIT_ING": ["1"],
        "NRO_TOTAL_HOSPIT_EGR": ["0"], "NRO_TOTAL_ESTANCIAS": ["0"],
        "NRO_TOTAL_PACIENTES_CAMAS": ["1"], "NRO_TOTAL_CAMAS": ["0"],
        "NRO_TOTAL_CAMAS_DISPONIB": ["1"], "NRO_TOTAL_FALLECIDOS": ["0"],
    })
    q05 = auditar_dataframe(fila).hallazgos.query("regla == 'Q05'").iloc[0]
    assert q05.codigo_ipress == "00009251"
    valores = json.loads(q05.valores_relevantes)
    assert valores["CO_IPRESS_ORIGINAL"] == "9251"
    assert valores["CO_IPRESS_CANONICO"] == "00009251"


def test_ruta_default_es_directorio_d1_dedicado_y_lectura_determinista(tmp_path):
    assert RAW_DATA_DIR.name == "Hospitalizacion"
    hospitalizacion = tmp_path / "Hospitalizacion"
    capacidad = tmp_path / "Capacidad"
    hospitalizacion.mkdir()
    capacidad.mkdir()
    (hospitalizacion / "b.csv").write_text("x\n1\n", encoding="utf-8")
    (hospitalizacion / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (capacidad / "ajeno.csv").write_text("x\n1\n", encoding="utf-8")
    assert [p.name for p in listar_archivos_csv(hospitalizacion)] == ["a.csv", "b.csv"]

