import json
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src import preparar_dataset as preparacion
from src.calidad_datos import COLUMNAS_HALLAZGOS, auditar_dataframe, auditar_directorio
from src.datos_raw import leer_csv


def fila(**cambios):
    datos = {
        "ANHO": "2016", "MES": "2", "UBIGEO": "150101",
        "DEPARTAMENTO": "LIMA", "PROVINCIA": "LIMA", "DISTRITO": "LIMA",
        "SECTOR": "MINSA", "CATEGORIA": "III-1", "CO_IPRESS": "00000001",
        "RAZON_SOC": "HOSPITAL DE PRUEBA", "ID_HOSPITALIZACION": "241500",
        "HOSPITALIZACION": "HOSPITALIZACION GENERAL", "NRO_TOTAL_HOSPIT_ING": "20",
        "NRO_TOTAL_HOSPIT_EGR": "18", "NRO_TOTAL_ESTANCIAS": "90",
        "NRO_TOTAL_PACIENTES_CAMAS": "1000", "NRO_TOTAL_CAMAS": "65",
        "NRO_TOTAL_CAMAS_DISPONIB": "1885", "NRO_TOTAL_FALLECIDOS": "0",
    }
    return {**datos, **cambios}


def auditar(**cambios):
    return auditar_dataframe(pd.DataFrame([fila(**cambios)]))


def reglas(resultado):
    return set(resultado.hallazgos["regla"])


@pytest.mark.parametrize("nombre", ["NRO_TOTAL_CAMAS_DISPONIB", "DIAS_CAMA_DISPONIBLE"])
def test_alias_y_febrero_bisiesto_normal(nombre):
    df = pd.DataFrame([fila()]).rename(columns={"NRO_TOTAL_CAMAS_DISPONIB": nombre})
    original = df.copy(deep=True)
    resultado = auditar_dataframe(df)
    assert resultado.resumen["estado_esquema"] == "OK"
    assert resultado.hallazgos.empty  # 65 × 29 = 1885 días-cama; no 1885 camas.
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("valor", ["abc", "NE_0001", "", "NA", "null", "inf", "-inf"])
def test_numerico_invalido_registra_literal_sin_cero(valor):
    resultado = auditar(NRO_TOTAL_HOSPIT_ING=valor)
    assert resultado.resumen["valores_numericos_invalidos"] == 1
    hallazgo = resultado.hallazgos.query("regla == 'Q01'").iloc[0]
    assert json.loads(hallazgo.valores_relevantes)["NRO_TOTAL_HOSPIT_ING"] == valor
    assert hallazgo.codigo_ipress == "00000001"
    assert hallazgo.fila_csv_aproximada == 2


@pytest.mark.parametrize("columna", preparacion.COLUMNAS_NUMERICAS)
def test_negativo_en_cada_variable(columna):
    resultado = auditar(**{columna: "-2"})
    assert resultado.resumen["valores_negativos"] == 1
    assert json.loads(resultado.hallazgos.query("regla == 'Q02'").iloc[0].valores_relevantes)[columna] == "-2"


@pytest.mark.parametrize("cambios", [
    {"MES": "0"}, {"MES": "13"}, {"MES": "2.5"}, {"MES": "abc"},
    {"ANHO": "0"}, {"ANHO": "2101"}, {"ANHO": "2016.5"}, {"ANHO": "inf"},
])
def test_periodos_invalidos_no_producen_alertas_de_calendario(cambios):
    resultado = auditar(**cambios)
    assert resultado.resumen["periodos_invalidos"] == 1
    assert not {"Q07", "Q09"} & reglas(resultado)


def test_duplicados_exactos_cuenta_repeticiones_no_elimina_ni_normaliza_valores():
    df = pd.DataFrame([fila(), fila(), fila(), fila(RAZON_SOC=" hospital de prueba ")])
    df.index = [3, 3, 7, 100]
    original = df.copy(deep=True)
    resultado = auditar_dataframe(df)
    assert resultado.resumen["duplicados_exactos"] == 2
    assert resultado.hallazgos.query("regla == 'Q04'").fila_csv_aproximada.tolist() == [3, 4]
    pd.testing.assert_frame_equal(df, original)


def test_cero_camas_con_actividad():
    assert "Q05" in reglas(auditar(NRO_TOTAL_CAMAS="0"))


def test_pacientes_dia_con_cero_dias_cama_error_no_division_por_cero():
    resultado = auditar(NRO_TOTAL_CAMAS_DISPONIB="0")
    assert resultado.hallazgos.query("regla == 'Q06'").iloc[0].severidad == "ERROR"
    assert "Q08" not in reglas(resultado)


def test_desplazamiento_fuerte_y_patron_cercano_no_confirmado():
    df = pd.DataFrame([fila(MES="12", NRO_TOTAL_PACIENTES_CAMAS="36", NRO_TOTAL_CAMAS="1116")])
    original = df.copy(deep=True)
    resultado = auditar_dataframe(df)
    assert "Q07" in reglas(resultado)
    pd.testing.assert_frame_equal(df, original)
    valores = json.loads(resultado.hallazgos.query("regla == 'Q07'").iloc[0].valores_relevantes)
    assert valores["NRO_TOTAL_CAMAS"] == "1116"
    assert valores["NRO_TOTAL_PACIENTES_CAMAS"] == "36"
    assert "Q07" not in reglas(auditar(MES="12", NRO_TOTAL_PACIENTES_CAMAS="36", NRO_TOTAL_CAMAS="1115"))


@pytest.mark.parametrize("pacientes, esperado", [("1200", False), ("1201", True)])
def test_ocupacion_auditada_usa_dias_cama_y_umbral_estricto(pacientes, esperado):
    resultado = auditar(NRO_TOTAL_PACIENTES_CAMAS=pacientes, NRO_TOTAL_CAMAS_DISPONIB="1000")
    assert ("Q08" in reglas(resultado)) == esperado
    if esperado:
        valor = json.loads(resultado.hallazgos.query("regla == 'Q08'").iloc[0].valores_relevantes)
        assert valor["ocupacion_auditada"] == pytest.approx(1.201)


@pytest.mark.parametrize("disponibles, esperado", [("1449", True), ("1450", False), ("2700", False), ("4350", False), ("4351", True)])
def test_q09_tolera_variacion_y_marca_extremos(disponibles, esperado):
    resultado = auditar(NRO_TOTAL_CAMAS="100", NRO_TOTAL_CAMAS_DISPONIB=disponibles)
    assert ("Q09" in reglas(resultado)) == esperado


def test_numericos_invalidos_no_simulan_ceros_en_reglas_cruzadas():
    resultado = auditar(NRO_TOTAL_CAMAS="error", NRO_TOTAL_CAMAS_DISPONIB="error")
    assert resultado.resumen["valores_numericos_invalidos"] == 2
    assert not {"Q05", "Q06", "Q07", "Q08", "Q09"} & reglas(resultado)


def test_esquema_incompleto_documenta_reglas_no_evaluadas():
    resultado = auditar_dataframe(pd.DataFrame([fila()]).drop(columns=["NRO_TOTAL_CAMAS"]))
    assert resultado.resumen["estado_esquema"] == "ERROR"
    assert resultado.resumen["filas_totales"] == 1
    assert "NRO_TOTAL_CAMAS" in resultado.resumen["detalle_esquema"]
    assert reglas(resultado) == {"Q00"}
    assert "Q09" in resultado.resumen["reglas_no_evaluadas"]


def test_ambiguedad_de_alias_o_columnas_no_se_resuelve_silenciosamente():
    assert "Q00" in reglas(auditar(DIAS_CAMA_DISPONIBLE="99"))
    assert "Q00" not in reglas(auditar(DIAS_CAMA_DISPONIBLE="1885"))
    df = pd.DataFrame([fila()])
    df[" mes "] = "2"
    assert "Q00" in reglas(auditar_dataframe(df))


def test_alcance_modelo_reutiliza_filtros_sin_ocultar_errores():
    df = pd.DataFrame([
        fila(NRO_TOTAL_CAMAS="0"),
        fila(DEPARTAMENTO="CUSCO", NRO_TOTAL_CAMAS="0"),
        fila(PROVINCIA="BARRANCA", NRO_TOTAL_CAMAS="0"),
        fila(SECTOR="PRIVADO", NRO_TOTAL_CAMAS="0"),
        fila(ID_HOSPITALIZACION="NE_0001", NRO_TOTAL_CAMAS="0"),
        fila(ID_HOSPITALIZACION="NE_0002", NRO_TOTAL_CAMAS="0"),
        fila(HOSPITALIZACION="  ", NRO_TOTAL_CAMAS="0"),
        fila(DEPARTAMENTO=" lima ", SECTOR=" minsa ", MES="13", NRO_TOTAL_CAMAS="0"),
    ])
    resultado = auditar_dataframe(df)
    assert resultado.resumen["filas_en_alcance_modelo"] == 2
    assert resultado.resumen["filas_en_alcance_modelo_periodo_valido"] == 1
    assert resultado.resumen["cero_camas_con_actividad"] == 8
    assert resultado.resumen["cero_camas_con_actividad_alcance_modelo"] == 2
    assert resultado.resumen["periodos_invalidos_alcance_modelo"] == 1


@pytest.mark.parametrize("encoding, separador", [("utf-8-sig", ","), ("latin1", ";")])
def test_reportes_encoding_separador_literales_e_inmutabilidad(tmp_path, encoding, separador):
    raw, quality = tmp_path / "raw", tmp_path / "quality"
    raw.mkdir()
    archivo = raw / "historia.csv"
    pd.DataFrame([fila(RAZON_SOC="HOSPITAL NIÑO", NRO_TOTAL_HOSPIT_ING="NA")]).to_csv(archivo, index=False, encoding=encoding, sep=separador)
    huella = sha256(archivo.read_bytes()).hexdigest()
    resultado = auditar_directorio(raw, quality)
    assert resultado.archivos_validos == [archivo]
    assert sha256(archivo.read_bytes()).hexdigest() == huella
    resumen = pd.read_csv(quality / "resumen_calidad.csv")
    assert resumen.iloc[0].encoding == encoding
    assert resumen.iloc[0].separador == separador
    hallazgos = pd.read_csv(quality / "hallazgos_calidad.csv", keep_default_na=False, dtype=str)
    assert list(hallazgos.columns) == COLUMNAS_HALLAZGOS
    assert json.loads(hallazgos.iloc[0].valores_relevantes)["NRO_TOTAL_HOSPIT_ING"] == "NA"
    assert hallazgos.iloc[0].nombre_ipress == "HOSPITAL NIÑO"
    metadata = json.loads((quality / "resumen_calidad.json").read_text(encoding="utf-8"))
    assert metadata["totales"]["valores_numericos_invalidos"] == 1
    assert leer_csv(archivo).NRO_TOTAL_HOSPIT_ING.isna().all()  # Compatibilidad de lectura anterior.
    assert leer_csv(archivo, conservar_originales=True).iloc[0].NRO_TOTAL_HOSPIT_ING == "NA"


def test_archivos_vacios_rotos_y_sin_esquema_no_ocultan_archivo_valido(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "vacio.csv").write_text("")
    (raw / "roto.csv").write_text('a,b\n"sin cierre,3')
    (raw / "incompleto.csv").write_text("a,b\n1,2\n")
    pd.DataFrame([fila()]).to_csv(raw / "valido.csv", index=False)
    resultado = auditar_directorio(raw, tmp_path / "quality")
    assert [p.name for p in resultado.archivos_validos] == ["valido.csv"]
    assert resultado.resumen["archivos_bloqueados"] == 3
    assert resultado.resumen["totales"]["errores_esquema"] == 3
    assert resultado.resumen["archivos_con_conteo_desconocido"] == 2


def test_reportes_sin_hallazgos_y_archivos_temporales(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([fila()]).to_csv(raw / "valido.csv", index=False)
    (raw / "~temporal.csv").write_text("ignorar")
    (raw / "nota.txt").write_text("ignorar")
    resultado = auditar_directorio(raw, tmp_path / "quality")
    assert resultado.resumen["cantidad_archivos"] == 1
    assert pd.read_csv(tmp_path / "quality/hallazgos_calidad.csv").empty


def test_prohibe_reportes_dentro_de_raw(tmp_path):
    with pytest.raises(ValueError, match="reportes no pueden"):
        auditar_directorio(tmp_path, tmp_path / "reportes")
    assert not (tmp_path / "reportes").exists()


def test_pipeline_guarda_evidencia_y_aparta_capacidad_pendiente(tmp_path, monkeypatch):
    raw, quality, processed = tmp_path / "raw", tmp_path / "quality", tmp_path / "processed"
    raw.mkdir()
    pd.DataFrame([
        fila(MES="1", NRO_TOTAL_HOSPIT_ING="texto", NRO_TOTAL_CAMAS="0", NRO_TOTAL_CAMAS_DISPONIB="0"),
        fila(MES="2"), fila(MES="3"),
    ]).to_csv(raw / "datos.csv", index=False)
    (raw / "esquema_malo.csv").write_text("a,b\n1,2\n")
    for nombre, valor in {"RAW_DATA_DIR": raw, "QUALITY_DIR": quality, "PROCESSED_DIR": processed,
                           "OUTPUT_PATH": processed / "dataset.csv", "DATASET_METADATA_PATH": processed / "metadata.json"}.items():
        monkeypatch.setattr(preparacion, nombre, valor)
    limpieza_original = preparacion.limpiar_registros

    def limpieza_con_evidencia(df):
        assert df.iloc[0].NRO_TOTAL_HOSPIT_ING == "texto"
        reporte = pd.read_csv(quality / "hallazgos_calidad.csv")
        assert {"Q00", "Q01", "Q05", "Q06", "Q09"}.issuperset(set(reporte.regla))
        assert {"Q00", "Q01", "Q05", "Q06"}.issubset(set(reporte.regla))
        limpio = limpieza_original(df)
        assert len(limpio) == 3
        assert limpio.iloc[0].NRO_TOTAL_HOSPIT_ING == 0
        return limpio

    monkeypatch.setattr(preparacion, "limpiar_registros", limpieza_con_evidencia)
    resultado = preparacion.preparar_dataset()
    assert resultado.periodo_actual.tolist() == ["2016-02"]
    assert resultado.periodo_predicho.tolist() == ["2016-03"]
    assert resultado.iloc[0].total_camas > 0
    assert list(resultado.columns) == preparacion.COLUMNAS_SALIDA


def test_pipeline_sin_esquemas_validos_deja_reportes_antes_de_fallar(tmp_path, monkeypatch):
    raw, quality = tmp_path / "raw", tmp_path / "quality"
    raw.mkdir()
    (raw / "incompleto.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(preparacion, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(preparacion, "QUALITY_DIR", quality)
    with pytest.raises(ValueError, match="ningún CSV"):
        preparacion.preparar_dataset()
    assert (quality / "resumen_calidad.json").is_file()


@pytest.mark.parametrize("caso,codigo", [("valido", 0), ("q05", 0), ("q00", 1)])
def test_cli_modulo_auditoria_con_csv_artificial(tmp_path, caso, codigo):
    raw, quality = tmp_path / "raw", tmp_path / "quality"
    raw.mkdir()
    archivo = raw / "artificial.csv"
    if caso == "q00":
        archivo.write_text("columna\nvalor\n")
    else:
        datos = fila(NRO_TOTAL_CAMAS="0") if caso == "q05" else fila()
        pd.DataFrame([datos]).to_csv(archivo, index=False)
    antes = archivo.read_bytes()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proceso = subprocess.run(
        [sys.executable, "-m", "src.calidad_datos", "--raw-dir", str(raw),
         "--output-dir", str(quality)],
        env=env, capture_output=True, timeout=60,
    )
    assert proceso.returncode == codigo, proceso.stderr.decode(errors="replace")
    assert archivo.read_bytes() == antes
    for nombre in ("resumen_calidad.csv", "hallazgos_calidad.csv", "resumen_calidad.json"):
        assert (quality / nombre).is_file()
    hallazgos = pd.read_csv(quality / "hallazgos_calidad.csv")
    if caso == "valido":
        assert hallazgos.empty
    else:
        assert caso.upper() in set(hallazgos.regla)
