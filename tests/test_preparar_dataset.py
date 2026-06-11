import pytest

from src.preparar_dataset import MENSAJE_CSV_VACIO, leer_csv


def test_leer_csv_rechaza_archivo_vacio(tmp_path) -> None:
    archivo = tmp_path / "dataset_vacio.csv"
    archivo.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="El archivo CSV original está vacío"):
        leer_csv(archivo)

    assert MENSAJE_CSV_VACIO.startswith("El archivo CSV original está vacío")


def test_leer_csv_rechaza_archivo_inexistente(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No se encontró"):
        leer_csv(tmp_path / "no_existe.csv")
