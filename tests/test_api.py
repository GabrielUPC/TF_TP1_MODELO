import pytest
from fastapi.testclient import TestClient

from src import main, predecir


client = TestClient(main.app)


def datos_base(anio: int = 2024, mes: int = 3) -> dict:
    return {
        "anio": anio,
        "mes": mes,
        "ubigeo": "150101",
        "departamento": "LIMA",
        "provincia": "LIMA",
        "distrito": "LIMA",
        "sector": "MINSA",
        "categoria_ipress": "III-1",
        "codigo_ipress": "00006207",
        "id_hospitalizacion": "241500",
        "servicio_hospitalizacion": "HOSPITALIZACION GENERAL",
        "total_ingresos": 80 + mes,
        "total_egresos": 70 + mes,
        "total_estancias": 350 + mes,
        "total_pacientes_camas": 2500 + mes,
        "total_camas": 100,
        "total_camas_disponibles": 3100,
        "total_fallecidos": 2,
    }


def prediccion_falsa(datos: dict) -> dict:
    return {
        "nivel_riesgo_predicho": "medio",
        "nivel_riesgo_codificado": 1,
        "probabilidad": 0.8,
        "probabilidades_por_clase": {
            "bajo": 0.1,
            "medio": 0.8,
            "alto": 0.1,
        },
        "variables_principales": [
            {"variable": "ocupacion_estimada", "valor": 0.81}
        ],
    }


def test_health() -> None:
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok"}


def test_cors_permite_origenes_de_desarrollo() -> None:
    respuesta = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:4200",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert respuesta.status_code == 200
    assert respuesta.headers["access-control-allow-origin"] == (
        "http://localhost:4200"
    )


def test_predict_devuelve_periodo_siguiente_y_calcula_indicadores(
    monkeypatch,
) -> None:
    registro_recibido = {}

    def capturar(datos: dict) -> dict:
        registro_recibido.update(datos)
        return prediccion_falsa(datos)

    monkeypatch.setattr(main, "predecir_riesgo", capturar)
    respuesta = client.post(
        "/predict",
        json={
            "registro_actual": datos_base(2024, 3),
            "historial_ultimos_meses": [
                datos_base(2024, 1),
                datos_base(2024, 2),
            ],
        },
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["periodo_actual"] == "2024-03"
    assert cuerpo["periodo_predicho"] == "2024-04"
    assert cuerpo["advertencia_historial"] is None
    assert registro_recibido["dias_mes"] == 31
    assert "promedio_movil_3m_ingresos" in registro_recibido
    assert registro_recibido["presion_ingresos_camas"] == pytest.approx(0.83)


def test_predict_sin_historial_devuelve_advertencia(monkeypatch) -> None:
    monkeypatch.setattr(main, "predecir_riesgo", prediccion_falsa)

    respuesta = client.post(
        "/predict",
        json={"registro_actual": datos_base(), "historial_ultimos_meses": []},
    )

    assert respuesta.status_code == 200
    assert respuesta.json()["advertencia_historial"]


def test_metadata_funciona(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "obtener_metadata_publica",
        lambda: {
            "tipo_modelo": "prediccion_siguiente_mes",
            "horizonte_prediccion": "mes_siguiente",
            "f1_macro_temporal": 0.75,
        },
    )

    respuesta = client.get("/metadata")

    assert respuesta.status_code == 200
    assert respuesta.json()["tipo_modelo"] == "prediccion_siguiente_mes"


def test_cargar_artefactos_informa_modelo_ausente(
    monkeypatch,
    tmp_path,
) -> None:
    predecir.limpiar_caches()
    monkeypatch.setattr(predecir, "MODEL_PATH", tmp_path / "ausente.joblib")

    with pytest.raises(FileNotFoundError, match="No se encontró el modelo"):
        predecir.cargar_artefactos()

    predecir.limpiar_caches()
