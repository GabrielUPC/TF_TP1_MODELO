import json

import pandas as pd
import pytest

from src.preparar_dataset import (
    MENSAJE_CSV_VACIO,
    crear_objetivo_futuro,
    crear_riesgo_actual,
    crear_variable_objetivo,
    leer_csv,
    leer_todos_los_csv,
)
from src.variables_temporales import agregar_variables_temporales


COLUMNAS_BASE = {
    "ANHO": ["2020"],
    "MES": ["1"],
    "UBIGEO": ["150101"],
    "DEPARTAMENTO": ["LIMA"],
    "PROVINCIA": ["LIMA"],
    "DISTRITO": ["LIMA"],
    "SECTOR": ["MINSA"],
    "CATEGORIA": ["III-1"],
    "CO_IPRESS": ["00000001"],
    "RAZON_SOC": ["IPRESS PRUEBA"],
    "ID_HOSPITALIZACION": ["241500"],
    "HOSPITALIZACION": ["HOSPITALIZACION GENERAL"],
    "NRO_TOTAL_HOSPIT_ING": ["10"],
    "NRO_TOTAL_HOSPIT_EGR": ["8"],
    "NRO_TOTAL_ESTANCIAS": ["40"],
    "NRO_TOTAL_PACIENTES_CAMAS": ["100"],
    "NRO_TOTAL_CAMAS": ["10"],
    "NRO_TOTAL_CAMAS_DISPONIB": ["310"],
    "NRO_TOTAL_FALLECIDOS": ["0"],
}


def test_leer_csv_rechaza_archivo_vacio(tmp_path) -> None:
    archivo = tmp_path / "dataset_vacio.csv"
    archivo.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="El archivo CSV original está vacío"):
        leer_csv(archivo)

    assert MENSAJE_CSV_VACIO.startswith("El archivo CSV original está vacío")


def test_leer_csv_rechaza_archivo_inexistente(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No se encontró"):
        leer_csv(tmp_path / "no_existe.csv")


def test_leer_todos_los_csv_combina_archivos_y_agrega_origen(tmp_path) -> None:
    pd.DataFrame(COLUMNAS_BASE).to_csv(
        tmp_path / "datos_2020.csv",
        index=False,
    )
    segundo = pd.DataFrame(COLUMNAS_BASE).assign(ANHO="2021")
    segundo = segundo.rename(
        columns={"NRO_TOTAL_CAMAS_DISPONIB": "DIAS_CAMA_DISPONIBLE"}
    )
    segundo.to_csv(tmp_path / "datos_2021.csv", index=False, sep=";")

    combinado = leer_todos_los_csv(tmp_path)

    assert len(combinado) == 2
    assert set(combinado["ARCHIVO_ORIGEN"]) == {
        "datos_2020.csv",
        "datos_2021.csv",
    }
    assert "NRO_TOTAL_CAMAS_DISPONIB" in combinado.columns


def test_leer_todos_los_csv_ignora_archivos_no_csv(tmp_path) -> None:
    pd.DataFrame(COLUMNAS_BASE).to_csv(tmp_path / "datos.csv", index=False)
    (tmp_path / "notas.txt").write_text("no debe leerse", encoding="utf-8")

    combinado = leer_todos_los_csv(tmp_path)

    assert combinado["ARCHIVO_ORIGEN"].tolist() == ["datos.csv"]


def _serie_riesgo(meses: list[int]) -> pd.DataFrame:
    riesgos = ["bajo", "medio", "alto"][: len(meses)]
    codigos = [0, 1, 2][: len(meses)]
    return pd.DataFrame(
        {
            "codigo_ipress": ["A"] * len(meses),
            "servicio_hospitalizacion": ["S"] * len(meses),
            "anio": [2024] * len(meses),
            "mes": meses,
            "nivel_riesgo_actual": riesgos,
            "nivel_riesgo_actual_codificado": codigos,
        }
    )


def test_crear_objetivo_futuro_desplaza_riesgo_un_mes() -> None:
    resultado = crear_objetivo_futuro(_serie_riesgo([1, 2, 3]))

    assert resultado["periodo_actual"].tolist() == ["2024-01", "2024-02"]
    assert resultado["periodo_predicho"].tolist() == ["2024-02", "2024-03"]
    assert resultado["nivel_riesgo_siguiente_mes"].tolist() == [
        "medio",
        "alto",
    ]


def test_crear_objetivo_futuro_elimina_ultimo_mes_y_no_salta_huecos() -> None:
    resultado = crear_objetivo_futuro(_serie_riesgo([1, 3]))

    assert resultado.empty


def test_variables_moviles_no_usan_informacion_futura() -> None:
    df = pd.DataFrame(
        {
            "codigo_ipress": ["A", "A", "A"],
            "servicio_hospitalizacion": ["S", "S", "S"],
            "anio": [2024, 2024, 2024],
            "mes": [1, 2, 3],
            "ocupacion_estimada": [0.4, 0.5, 9.0],
            "presion_ingresos_camas": [0.2, 0.3, 9.0],
            "total_ingresos": [10, 20, 1000],
            "total_egresos": [8, 18, 900],
            "total_estancias": [40, 90, 5000],
        }
    )

    resultado = agregar_variables_temporales(df)

    assert resultado.loc[0, "promedio_movil_3m_ingresos"] == pytest.approx(10)
    assert resultado.loc[1, "promedio_movil_3m_ingresos"] == pytest.approx(15)
    assert resultado.loc[1, "tendencia_ingresos_1m"] == pytest.approx(10)


@pytest.mark.parametrize("ocupacion,nivel,codigo", [
    (0.69, "bajo", 0), (0.70, "medio", 1), (0.8499, "medio", 1),
    (0.85, "alto", 2), (1.00, "alto", 2),
])
def test_riesgo_actual_solo_requiere_ocupacion_y_respeta_limites(ocupacion, nivel, codigo):
    df = pd.DataFrame({"ocupacion_estimada": [ocupacion]}, index=[17])
    original = df.copy(deep=True)
    resultado, percentiles = crear_riesgo_actual(df)
    assert resultado.loc[17, "nivel_riesgo_actual"] == nivel
    assert resultado.loc[17, "nivel_riesgo_actual_codificado"] == codigo
    assert percentiles == {}
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("ocupacion,nivel", [(0.69, "bajo"), (0.70, "medio"), (0.85, "alto")])
@pytest.mark.parametrize("columna,valores", [
    ("ratio_camas_disponibles", [0.0, 0.10, 0.20, 1.0, 5.0]),
    ("presion_ingresos_camas", [0.0, 1.0, 100.0, 10000.0, 1000000.0]),
])
def test_indicadores_auxiliares_no_cambian_etiqueta(ocupacion, nivel, columna, valores):
    df = pd.DataFrame({"ocupacion_estimada": [ocupacion]*len(valores),
                       "ratio_camas_disponibles": 1.0, "presion_ingresos_camas": 0.0})
    df[columna] = valores
    resultado, _ = crear_riesgo_actual(df)
    assert resultado.nivel_riesgo_actual.tolist() == [nivel]*len(valores)
    pd.testing.assert_frame_equal(resultado[df.columns], df)


def test_etiqueta_no_depende_de_distribucion_ni_percentiles_legados(monkeypatch):
    def prohibir_percentiles(*args, **kwargs):
        pytest.fail("La etiqueta no debe calcular percentiles globales")
    monkeypatch.setattr(pd.Series, "quantile", prohibir_percentiles)
    base = pd.DataFrame({"ocupacion_estimada": [0.69, 0.70, 0.85],
                         "ratio_camas_disponibles": [0.0, 1.0, 2.0],
                         "presion_ingresos_camas": [0.0, 5.0, 10.0]})
    extendido = pd.concat([base, base.assign(presion_ingresos_camas=1e12)], ignore_index=True)
    primero, _ = crear_riesgo_actual(base)
    segundo, percentiles = crear_riesgo_actual(extendido, percentiles={
        "presion_ingresos_camas_percentil_50": -1,
        "presion_ingresos_camas_percentil_75": -1,
    })
    pd.testing.assert_frame_equal(primero, segundo.iloc[:len(base)])
    assert percentiles == {}


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), -0.1, None])
def test_ocupacion_invalida_no_se_etiqueta_como_bajo(valor):
    with pytest.raises(ValueError, match="finita y no negativa"):
        crear_riesgo_actual(pd.DataFrame({"ocupacion_estimada": [valor]}))


def test_nuevo_objetivo_respeta_grupos_cambio_anio_y_huecos():
    filas = []
    for ipress, servicio, ocupaciones in [
        ("A", "S", [0.69, 0.70, 0.85, 1.0]),
        ("B", "S", [0.85, 0.69, 0.70, 0.85]),
        ("A", "T", [0.70, 0.85, 0.69, 0.70]),
    ]:
        for (anio, mes), ocupacion in zip([(2023, 12), (2024, 1), (2024, 3), (2024, 4)], ocupaciones):
            filas.append({"codigo_ipress": ipress, "servicio_hospitalizacion": servicio,
                          "anio": anio, "mes": mes, "ocupacion_estimada": ocupacion})
    df = pd.DataFrame(filas).sample(frac=1, random_state=7)
    resultado = crear_variable_objetivo(df)
    assert len(resultado) == 6
    assert set(resultado.periodo_actual) == {"2023-12", "2024-03"}
    assert set(resultado.periodo_predicho) == {"2024-01", "2024-04"}
    esperados = {("A", "S"): ["medio", "alto"], ("B", "S"): ["bajo", "alto"],
                 ("A", "T"): ["alto", "medio"]}
    for clave, niveles in esperados.items():
        grupo = resultado[(resultado.codigo_ipress == clave[0]) &
                          (resultado.servicio_hospitalizacion == clave[1])].sort_values("periodo_actual")
        assert grupo.nivel_riesgo_siguiente_mes.tolist() == niveles
        assert grupo.nivel_riesgo_siguiente_mes_codificado.tolist() == [
            {"bajo": 0, "medio": 1, "alto": 2}[nivel] for nivel in niveles]


def test_pipeline_metadata_documenta_etiqueta_observada_sin_percentiles(tmp_path, monkeypatch):
    from src import preparar_dataset as prep
    raw, quality, processed = [tmp_path/n for n in ("raw", "quality", "processed")]
    raw.mkdir()
    dataset, metadata = processed/"dataset.csv", processed/"metadata.json"
    for nombre, valor in {"RAW_DATA_DIR": raw, "QUALITY_DIR": quality, "PROCESSED_DIR": processed,
                          "OUTPUT_PATH": dataset, "DATASET_METADATA_PATH": metadata}.items():
        monkeypatch.setattr(prep, nombre, valor)
    filas = [pd.DataFrame(COLUMNAS_BASE).assign(MES=str(mes), NRO_TOTAL_CAMAS="4",
             NRO_TOTAL_CAMAS_DISPONIB="100", NRO_TOTAL_PACIENTES_CAMAS=str(pacientes))
             for mes, pacientes in [(1, 60), (2, 70), (3, 85)]]
    fuente = raw/"artificial.csv"
    pd.concat(filas, ignore_index=True).to_csv(fuente, index=False)
    antes = fuente.read_bytes()
    resultado = prep.preparar_dataset()
    assert fuente.read_bytes() == antes
    assert resultado.nivel_riesgo_actual.tolist() == ["bajo", "medio"]
    assert resultado.nivel_riesgo_siguiente_mes.tolist() == ["medio", "alto"]
    assert {"ratio_camas_disponibles", "presion_ingresos_camas"}.issubset(resultado.columns)
    info = json.loads(metadata.read_text(encoding="utf-8"))
    assert info["percentiles_riesgo_actual"] == {}
    assert info["metodo_percentiles"] == "No se utilizan percentiles para construir la etiqueta."
    definicion = info["definicion_target"]
    assert definicion["version"] == "riesgo_ocupacion_observada_v2"
    assert definicion["nivel_riesgo_actual"] == {
        "bajo": "ocupacion_estimada < 0.70",
        "medio": "0.70 <= ocupacion_estimada < 0.85",
        "alto": "ocupacion_estimada >= 0.85",
    }
    assert definicion["codificacion"] == {"bajo": 0, "medio": 1, "alto": 2}
    assert not definicion["usa_ratio_camas_disponibles"]
    assert not definicion["usa_percentiles_presion_ingresos_camas"]
