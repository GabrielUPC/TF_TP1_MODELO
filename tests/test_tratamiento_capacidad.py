from calendar import monthrange
from datetime import datetime
import json

import pandas as pd
import pytest

from src import preparar_dataset as prep
from src.tratamiento_capacidad import sha256_archivo


def fila(mes, **cambios):
    return {
        "ANHO": "2024", "MES": str(mes), "UBIGEO": "150101",
        "DEPARTAMENTO": "LIMA", "PROVINCIA": "LIMA", "DISTRITO": "LIMA",
        "SECTOR": "MINSA", "CATEGORIA": "III-1", "CO_IPRESS": "00000001",
        "RAZON_SOC": "HOSPITAL PRUEBA", "ID_HOSPITALIZACION": "241500",
        "HOSPITALIZACION": "MEDICINA GENERAL", "NRO_TOTAL_HOSPIT_ING": str(mes*10),
        "NRO_TOTAL_HOSPIT_EGR": "8", "NRO_TOTAL_ESTANCIAS": "40",
        "NRO_TOTAL_PACIENTES_CAMAS": "100", "NRO_TOTAL_CAMAS": "10",
        "NRO_TOTAL_CAMAS_DISPONIB": str(10*monthrange(2024, mes)[1]),
        "NRO_TOTAL_FALLECIDOS": "0",
    } | cambios


def rutas(tmp_path, monkeypatch):
    raw, quality, processed = [tmp_path/n for n in ("raw", "quality", "processed")]
    raw.mkdir()
    dataset, metadata = processed/"dataset.csv", processed/"metadata.json"
    for nombre, valor in {
        "RAW_DATA_DIR": raw, "QUALITY_DIR": quality, "PROCESSED_DIR": processed,
        "OUTPUT_PATH": dataset, "DATASET_METADATA_PATH": metadata,
    }.items():
        monkeypatch.setattr(prep, nombre, valor)
    return raw, quality, dataset, metadata


def fila_q07(mes=2, **cambios):
    dias = monthrange(2024, mes)[1]
    # Q07 aislado: no cumple Q05, Q06, Q08 ni Q09.
    return fila(mes, **({"NRO_TOTAL_PACIENTES_CAMAS": "7",
                        "NRO_TOTAL_CAMAS": str(7*dias),
                        "NRO_TOTAL_CAMAS_DISPONIB": str(7*dias*dias)} | cambios))


@pytest.mark.parametrize("regla", ["Q05", "Q06", "Q07"])
def test_aparta_grupo_antes_de_consolidar_y_no_salta_huecos(tmp_path, monkeypatch, regla):
    raw, quality, dataset, metadata = rutas(tmp_path, monkeypatch)
    filas = [fila(mes) for mes in range(1, 7)]
    filas[0]["NRO_TOTAL_HOSPIT_ING"] = "900"
    invalida = fila_q07() if regla == "Q07" else fila(2)
    if regla != "Q07":
        invalida["NRO_TOTAL_CAMAS" if regla == "Q05" else "NRO_TOTAL_CAMAS_DISPONIB"] = "0"
    invalida.update(HOSPITALIZACION=" medicina   general ", RAZON_SOC="OTRO NOMBRE",
                    NRO_TOTAL_HOSPIT_ING="9999")
    filas.append(invalida)
    fuente = raw/"serie.csv"
    pd.DataFrame(filas).to_csv(fuente, index=False)
    antes = fuente.read_bytes()

    resultado = prep.preparar_dataset()

    assert fuente.read_bytes() == antes
    assert resultado.periodo_actual.tolist() == ["2024-03", "2024-04", "2024-05"]
    assert resultado.periodo_predicho.tolist() == ["2024-04", "2024-05", "2024-06"]
    assert resultado.iloc[0].tendencia_ingresos_1m == 0
    assert resultado.promedio_movil_3m_ingresos.tolist() == [30, 35, 40]
    info = json.loads(metadata.read_text())
    politica = info["tratamiento_capacidad"]
    assert politica["version"] == "capacidad_q05_q06_q07_v2"
    assert politica["reglas_aplicadas"] == ["Q05", "Q06", "Q07"]
    assert politica["reglas_solo_auditadas"] == ["Q01", "Q02", "Q03", "Q04", "Q08", "Q09"]
    assert politica["filas_raw_pendientes"] == 1
    assert politica["meses_servicio_pendientes"] == 1
    assert politica["filas_apartadas_antes_consolidacion"] == 2
    assert datetime.fromisoformat(politica["generado_utc"]).tzinfo is not None
    assert politica == json.loads((quality/"tratamiento_capacidad.json").read_text())
    assert info["raw_sha256"] == politica["raw_sha256"] == {"serie.csv": sha256_archivo(fuente)}
    assert info["dataset_sha256"] == sha256_archivo(dataset)
    assert politica["auditoria_sha256"] == sha256_archivo(quality/"hallazgos_calidad.csv")
    pendientes = pd.read_csv(quality/"pendientes_capacidad.csv", dtype=str)
    assert regla in set(pendientes.regla)
    assert set(pendientes.estado_revision) == {"PENDIENTE_VALIDACION_NO_USAR_ENTRENAMIENTO"}
    assert set(pendientes.codigo_ipress) == {"00000001"}
    if regla == "Q07":
        assert set(pd.read_csv(quality/"hallazgos_calidad.csv").regla) == {"Q07"}


@pytest.mark.parametrize("regla,cambios", [
    ("Q08", {"NRO_TOTAL_PACIENTES_CAMAS": "400"}),
    ("Q09", {"NRO_TOTAL_CAMAS_DISPONIB": "100"}),
])
def test_q08_y_q09_no_apartan_grupos(tmp_path, monkeypatch, regla, cambios):
    raw, quality, _, metadata = rutas(tmp_path, monkeypatch)
    pd.DataFrame([fila(1, **cambios), fila(2), fila(3)]).to_csv(raw/"serie.csv", index=False)
    resultado = prep.preparar_dataset()
    assert resultado.periodo_actual.tolist() == ["2024-01", "2024-02"]
    assert set(pd.read_csv(quality/"hallazgos_calidad.csv").regla) == {regla}
    assert json.loads(metadata.read_text())["tratamiento_capacidad"]["meses_servicio_pendientes"] == 0
    assert pd.read_csv(quality/"pendientes_capacidad.csv").empty


@pytest.mark.parametrize("otra_clave", [
    {"CO_IPRESS": "00000002"}, {"HOSPITALIZACION": "CIRUGIA GENERAL"},
])
def test_no_aparta_otra_ipress_o_servicio_del_mismo_mes(tmp_path, monkeypatch, otra_clave):
    raw, _, _, _ = rutas(tmp_path, monkeypatch)
    filas = [fila(mes) for mes in range(1, 4)]
    filas += [fila(mes, **otra_clave) for mes in range(1, 4)]
    filas.append(fila_q07())
    pd.DataFrame(filas).to_csv(raw/"serie.csv", index=False)
    resultado = prep.preparar_dataset()
    assert resultado.periodo_actual.tolist() == ["2024-01", "2024-02"]
    if "CO_IPRESS" in otra_clave:
        assert set(resultado.codigo_ipress) == {"00000002"}
    else:
        assert set(resultado.servicio_hospitalizacion) == {"CIRUGIA GENERAL"}


def test_q07_fuera_de_alcance_se_audita_pero_no_aparta_el_grupo_local(tmp_path, monkeypatch):
    raw, quality, _, metadata = rutas(tmp_path, monkeypatch)
    pd.DataFrame([fila(1), fila(2), fila(3),
                  fila_q07(DEPARTAMENTO="CUSCO", PROVINCIA="CUSCO")]).to_csv(raw/"serie.csv", index=False)
    resultado = prep.preparar_dataset()
    assert resultado.periodo_actual.tolist() == ["2024-01", "2024-02"]
    hallazgos = pd.read_csv(quality/"hallazgos_calidad.csv")
    assert set(hallazgos.regla) == {"Q07"}
    assert not hallazgos.en_alcance_modelo.any()
    assert json.loads(metadata.read_text())["tratamiento_capacidad"]["meses_servicio_pendientes"] == 0


def test_sin_parejas_no_sobrescribe_dataset_anterior(tmp_path, monkeypatch):
    raw, _, dataset, _ = rutas(tmp_path, monkeypatch)
    dataset.parent.mkdir()
    dataset.write_text("dataset anterior")
    huella = sha256_archivo(dataset)
    pd.DataFrame([fila(1), fila_q07(), fila(3)]).to_csv(raw/"serie.csv", index=False)
    with pytest.raises(ValueError, match="No quedan parejas"):
        prep.preparar_dataset()
    assert sha256_archivo(dataset) == huella
