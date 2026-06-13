import pandas as pd
import pytest

from src.preparar_dataset import (
    MENSAJE_CSV_VACIO,
    crear_objetivo_futuro,
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
