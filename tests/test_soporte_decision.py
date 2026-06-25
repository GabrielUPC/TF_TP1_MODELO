from src.soporte_decision import (
    calcular_brecha_operativa,
    causa_principal_riesgo,
    generar_soporte_decision,
)


def datos_operativos(**overrides: float) -> dict[str, float]:
    datos = {
        "ocupacion_estimada": 0.91,
        "presion_ingresos_camas": 1.2,
        "promedio_estancia": 8.0,
        "rotacion_camas": 0.8,
        "diferencia_ingresos_egresos": 5.0,
        "ratio_camas_disponibles": 0.08,
        "total_ingresos": 90.0,
        "total_egresos": 85.0,
        "total_camas": 75.0,
        "total_camas_disponibles": 210.0,
    }
    datos.update(overrides)
    return datos


def test_causa_prioriza_ocupacion_critica() -> None:
    assert causa_principal_riesgo(datos_operativos(), "alto") == "Ocupación crítica"


def test_brecha_operativa_limita_puntaje_y_clasifica_critica() -> None:
    brecha = calcular_brecha_operativa(datos_operativos(), "alto")

    assert brecha["brecha_operativa"] == 100
    assert brecha["nivel_brecha_operativa"] == "Brecha crítica"


def test_soporte_decision_incluye_recomendaciones_accionables() -> None:
    soporte = generar_soporte_decision(
        datos_operativos(ocupacion_estimada=0.75, ratio_camas_disponibles=0.5),
        "medio",
        0.72,
    )

    assert soporte["causa_principal_riesgo"] == "Demanda supera egresos"
    assert soporte["recomendaciones_operativas"]
    assert soporte["confianza_prediccion"] == 0.72
