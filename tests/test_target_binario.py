"""Contrato científico del target binario; datos únicamente artificiales."""

import json

import numpy as np
import pandas as pd
import pytest

from src import preparar_dataset as prep


def _serie(ocupaciones, *, ipress="00000001", servicio="MEDICINA", inicio="2024-01"):
    periodos = pd.period_range(inicio, periods=len(ocupaciones), freq="M")
    return pd.DataFrame({
        "codigo_ipress": ipress,
        "servicio_hospitalizacion": servicio,
        "anio": periodos.year,
        "mes": periodos.month,
        "ocupacion_estimada": ocupaciones,
    })


@pytest.mark.parametrize("ocupacion_futura,clase", [
    (0.0, 0), (0.85, 0), (0.899999, 0), (0.90, 0), (0.900001, 1), (1.0, 1),
])
def test_target_binario_comparacion_estricta_del_mes_siguiente(ocupacion_futura, clase):
    entrada = _serie([1.1 if clase == 0 else 0.2, ocupacion_futura])
    resultado = prep.crear_objetivo_futuro(entrada)
    assert resultado["ocupacion_siguiente_mes"].tolist() == [ocupacion_futura]
    assert resultado[prep.TARGET_BINARIO].tolist() == [clase]
    assert resultado["periodo_actual"].tolist() == ["2024-01"]
    assert resultado["periodo_predicho"].tolist() == ["2024-02"]


def test_target_binario_separa_ipress_y_servicio_y_conserva_orden_de_entrada():
    entrada = pd.concat([
        _serie([1.0, 0.4], ipress="00000001", servicio="MEDICINA"),
        _serie([0.1, 1.0], ipress="00000002", servicio="MEDICINA"),
        _serie([0.1, 0.900001], ipress="00000001", servicio="CIRUGIA"),
    ], ignore_index=True).iloc[[4, 3, 0, 5, 2, 1]].copy()
    entrada.index = [901, 21, 55, 4, 880, 37]
    original = entrada.copy(deep=True)

    resultado = prep.crear_objetivo_futuro(entrada)

    assert list(zip(resultado.codigo_ipress, resultado.servicio_hospitalizacion)) == [
        ("00000001", "CIRUGIA"), ("00000001", "MEDICINA"), ("00000002", "MEDICINA"),
    ]
    assert resultado[prep.TARGET_BINARIO].tolist() == [1, 0, 1]
    assert resultado.ocupacion_siguiente_mes.tolist() == [0.900001, 0.4, 1.0]
    pd.testing.assert_frame_equal(entrada, original)


def test_target_binario_no_inventa_pares_entre_series_incompletas():
    entrada = pd.concat([
        _serie([0.8], ipress="00000001", servicio="MEDICINA", inicio="2024-01"),
        _serie([1.0], ipress="00000002", servicio="MEDICINA", inicio="2024-02"),
        _serie([1.0], ipress="00000001", servicio="CIRUGIA", inicio="2024-02"),
    ], ignore_index=True)
    assert prep.crear_objetivo_futuro(entrada).empty


def test_target_rechaza_ocupacion_nullable_ausente():
    entrada = _serie([0.8, 0.9])
    entrada["ocupacion_estimada"] = entrada["ocupacion_estimada"].astype("Float64")
    entrada.loc[1, "ocupacion_estimada"] = pd.NA
    with pytest.raises(ValueError, match="finitas y no negativas"):
        prep.crear_objetivo_futuro(entrada)


def test_target_binario_cruza_diciembre_pero_no_salta_mes_faltante():
    entrada = _serie([0.2, 1.0, 0.3, 0.9, 0.95], inicio="2023-12")
    # Febrero apartado: enero no puede utilizar marzo como t+1.
    entrada = entrada.loc[entrada.mes.ne(2)]
    resultado = prep.crear_objetivo_futuro(entrada)
    assert resultado.periodo_actual.tolist() == ["2023-12", "2024-03"]
    assert resultado.periodo_predicho.tolist() == ["2024-01", "2024-04"]
    assert resultado[prep.TARGET_BINARIO].tolist() == [1, 1]


def test_target_binario_rechaza_duplicados_de_un_mes():
    serie = _serie([0.5, 1.0])
    entrada = pd.concat([serie, serie.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicados"):
        prep.crear_objetivo_futuro(entrada)


@pytest.mark.parametrize("ocupacion", [np.nan, np.inf, -np.inf, -0.01, None, "invalida"])
def test_target_binario_no_etiqueta_observacion_invalida_como_cero(ocupacion):
    with pytest.raises((ValueError, TypeError)):
        prep.crear_objetivo_futuro(_serie([0.5, ocupacion]))


def test_target_binario_independiente_de_etiquetas_legacy_y_de_indicadores_auxiliares():
    base = _serie([0.2, 0.90, 0.900001])
    contaminada = base.assign(
        nivel_riesgo_actual=["alto", "alto", "bajo"],
        nivel_riesgo_actual_codificado=[2, 2, 0],
        ratio_camas_disponibles=[0.0, 99.0, 0.0],
        presion_ingresos_camas=[1e9, 0.0, 1e12],
    )
    sin_legacy = prep.crear_objetivo_futuro(base)
    con_legacy = prep.crear_objetivo_futuro(contaminada)
    assert sin_legacy[prep.TARGET_BINARIO].tolist() == [0, 1]
    pd.testing.assert_series_equal(sin_legacy[prep.TARGET_BINARIO], con_legacy[prep.TARGET_BINARIO])
    assert not set(prep.COLUMNAS_OBJETIVO_LEGACY).intersection(sin_legacy.columns)


def test_preparacion_binaria_salida_externa_metadata_e_integridad_raw(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    externa = tmp_path / "experimento"
    oficial = tmp_path / "oficial"
    oficial.mkdir()
    dataset_oficial = oficial / "dataset.csv"
    metadata_oficial = oficial / "metadata.json"
    dataset_oficial.write_bytes(b"dataset protegido\n")
    metadata_oficial.write_bytes(b'{"modelo":"legacy"}\n')
    # Capacidad/actividad consistentes, sin Q05-Q08. Ocupaciones de destino:
    # 0.90 exacto y 0.91, para comprobar ambos lados del umbral en el pipeline.
    filas = []
    for mes, pacientes in [(1, 250), (2, 270), (3, 273)]:
        filas.append({
            "ANHO": "2024", "MES": str(mes), "UBIGEO": "150101",
            "DEPARTAMENTO": "LIMA", "PROVINCIA": "LIMA", "DISTRITO": "LIMA",
            "SECTOR": "MINSA", "CATEGORIA": "III-1", "CO_IPRESS": "9251",
            "RAZON_SOC": "IPRESS ARTIFICIAL", "ID_HOSPITALIZACION": "241800",
            "HOSPITALIZACION": "MEDICINA ARTIFICIAL", "NRO_TOTAL_HOSPIT_ING": "12",
            "NRO_TOTAL_HOSPIT_EGR": "10", "NRO_TOTAL_ESTANCIAS": "40",
            "NRO_TOTAL_PACIENTES_CAMAS": str(pacientes), "NRO_TOTAL_CAMAS": "10",
            "NRO_TOTAL_CAMAS_DISPONIB": "300", "NRO_TOTAL_FALLECIDOS": "0",
        })
    fuente = raw / "artificial.csv"
    pd.DataFrame(filas).to_csv(fuente, index=False)
    huellas_antes = {p: p.read_bytes() for p in (fuente, dataset_oficial, metadata_oficial)}
    monkeypatch.setattr(prep, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(prep, "OUTPUT_PATH", dataset_oficial)
    monkeypatch.setattr(prep, "DATASET_METADATA_PATH", metadata_oficial)
    monkeypatch.setattr(prep, "QUALITY_DIR", oficial / "quality")
    monkeypatch.setattr(prep, "PROCESSED_DIR", oficial)

    resultado = prep.preparar_dataset(output_dir=externa)

    assert resultado[prep.TARGET_BINARIO].tolist() == [0, 1]
    assert resultado.ocupacion_siguiente_mes.tolist() == [0.9, 0.91]
    assert resultado.codigo_ipress.tolist() == ["00009251", "00009251"]
    assert resultado.familia_servicio.tolist() == ["hospitalizacion", "hospitalizacion"]
    assert all(p.read_bytes() == contenido for p, contenido in huellas_antes.items())
    assert not (oficial / "quality").exists()
    assert (externa / "quality" / "tratamiento_capacidad.json").is_file()
    assert len(pd.read_csv(externa / "meses_antes_target.csv")) == 3
    info = json.loads((externa / "dataset_metadata.json").read_text(encoding="utf-8"))
    target = info["definicion_target"]
    assert target["version"] == "alta_presion_capacidad_t1_v1"
    assert target["tipo"] == "clasificacion_binaria"
    assert target["variable_objetivo"] == prep.TARGET_BINARIO
    assert target["indicador"] == "ocupacion_siguiente_mes"
    assert target["comparador"] == ">" and target["umbral"] == 0.90
    assert target["numero_total"] == 2
    assert target["positivos"] == 1 and target["negativos"] == 1
    assert target["prevalencia"] == pytest.approx(0.5)
    assert info["numero_filas_antes_objetivo_futuro"] == 3
    assert info["numero_filas_entrenamiento"] == 2
    assert "sin saltar huecos" in target["regla_continuidad"]
    assert "Solo compatibilidad" in info["etiquetas_legacy"]["estado"]


@pytest.mark.parametrize("modulo,funcion", [("entrenar_modelo", "separar_variables"),
                                           ("backtesting_temporal", "validar_periodos")])
def test_motor_legacy_rechaza_dataset_binario(modulo, funcion):
    import importlib
    motor = importlib.import_module("src." + modulo)
    with pytest.raises(ValueError, match="binario"):
        getattr(motor, funcion)(pd.DataFrame({prep.TARGET_BINARIO: [0, 1]}))
