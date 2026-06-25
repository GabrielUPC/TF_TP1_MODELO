from src.interpretacion import (
    factores_explicativos_riesgo,
    formatear_ocupacion,
    interpretar_semaforo,
)


def test_formatear_ocupacion_convierte_ratio_a_porcentaje() -> None:
    assert formatear_ocupacion(0.85) == "85.0%"
    assert formatear_ocupacion(1.37) == "137.0%"
    assert formatear_ocupacion(1.4) == "140.0%"


def test_semaforo_alto_incluye_recomendacion_referencial() -> None:
    resultado = interpretar_semaforo("alto")

    assert resultado["color_semaforo"] == "rojo"
    assert "posible insuficiencia" in resultado["interpretacion_riesgo"]
    assert (
        "El resultado es referencial. No asigna camas automáticamente "
        "y no reemplaza decisiones clínicas."
    ) in resultado["recomendacion_riesgo"]


def test_factores_alto_explican_presion_y_contexto_del_modelo() -> None:
    factores = factores_explicativos_riesgo(
        {
            "ocupacion_estimada": 1.05,
            "total_pacientes_camas": 1050,
            "total_camas_disponibles": 1000,
            "promedio_estancia": 8,
            "presion_ingresos_camas": 1.2,
        },
        "alto",
    )

    texto = " ".join(factores)
    assert "supera la capacidad mensual registrada" in texto
    assert "estancias hospitalarias" in texto
    assert "histórico, tendencias" in texto
