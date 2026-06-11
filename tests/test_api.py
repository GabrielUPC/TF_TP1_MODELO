import pytest
from fastapi.testclient import TestClient

from src import main, predecir


client = TestClient(main.app)


def datos_base() -> dict:
    return {
        "anio": 2015,
        "mes": 1,
        "ubigeo": "150101",
        "departamento": "LIMA",
        "provincia": "LIMA",
        "distrito": "LIMA",
        "sector": "MINSA",
        "categoria_ipress": "III-1",
        "codigo_ipress": "00006207",
        "id_hospitalizacion": "241500",
        "servicio_hospitalizacion": "HOSPITALIZACION GENERAL",
        "total_ingresos": 80,
        "total_egresos": 70,
        "total_estancias": 350,
        "total_pacientes_camas": 2500,
        "total_camas": 100,
        "total_camas_disponibles": 3100,
        "total_fallecidos": 2,
    }


def test_health() -> None:
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok"}


def test_predict_calcula_indicadores_internamente(monkeypatch) -> None:
    registro_recibido = {}

    def predecir_falso(datos: dict) -> dict:
        registro_recibido.update(datos)
        return {
            "nivel_riesgo": "medio",
            "nivel_riesgo_codificado": 1,
            "probabilidad": 0.8,
            "probabilidades_por_clase": {
                "bajo": 0.1,
                "medio": 0.8,
                "alto": 0.1,
            },
        }

    monkeypatch.setattr(main, "predecir_riesgo", predecir_falso)
    respuesta = client.post("/predict", json=datos_base())

    assert respuesta.status_code == 200
    assert registro_recibido["dias_mes"] == 31
    assert registro_recibido["ratio_camas_disponibles"] == pytest.approx(1.0)
    assert registro_recibido["presion_ingresos_camas"] == pytest.approx(0.8)


def test_cargar_artefactos_informa_modelo_ausente(
    monkeypatch,
    tmp_path,
) -> None:
    predecir.cargar_artefactos.cache_clear()
    monkeypatch.setattr(predecir, "MODEL_PATH", tmp_path / "modelo_ausente.joblib")

    with pytest.raises(FileNotFoundError, match="No se encontró el modelo"):
        predecir.cargar_artefactos()

    predecir.cargar_artefactos.cache_clear()
