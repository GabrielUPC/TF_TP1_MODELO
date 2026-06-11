import pytest
import pandas as pd

from src.entrenar_modelo import crear_particion_temporal
from src.preparar_dataset import (
    MENSAJE_CSV_VACIO,
    leer_csv,
    leer_todos_los_csv,
)


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
    segundo.to_csv(
        tmp_path / "datos_2021.csv",
        index=False,
        sep=";",
    )

    combinado = leer_todos_los_csv(tmp_path)

    assert len(combinado) == 2
    assert set(combinado["ARCHIVO_ORIGEN"]) == {
        "datos_2020.csv",
        "datos_2021.csv",
    }
    assert "NRO_TOTAL_CAMAS_DISPONIB" in combinado.columns


def test_leer_todos_los_csv_ignora_archivos_no_csv(tmp_path) -> None:
    pd.DataFrame(COLUMNAS_BASE).to_csv(
        tmp_path / "datos.csv",
        index=False,
    )
    (tmp_path / "notas.txt").write_text("no debe leerse", encoding="utf-8")

    combinado = leer_todos_los_csv(tmp_path)

    assert combinado["ARCHIVO_ORIGEN"].tolist() == ["datos.csv"]


def test_evaluacion_temporal_se_omite_con_un_anio(capsys) -> None:
    df = pd.DataFrame(
        {
            "anio": [2020, 2020, 2020],
            "variable": [1, 2, 3],
            "nivel_riesgo_codificado": [0, 1, 2],
        }
    )
    X = df[["anio", "variable"]]
    y = df["nivel_riesgo_codificado"]

    resultado = crear_particion_temporal(df, X, y)

    assert resultado is None
    assert "solo hay un año disponible" in capsys.readouterr().out
