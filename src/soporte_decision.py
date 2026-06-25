from __future__ import annotations

from typing import Any


INDICADORES_OPERATIVOS = [
    "ocupacion_estimada",
    "presion_ingresos_camas",
    "promedio_estancia",
    "rotacion_camas",
    "diferencia_ingresos_egresos",
    "ratio_camas_disponibles",
    "total_ingresos",
    "total_egresos",
    "total_camas",
    "total_camas_disponibles",
]


def valor_numerico(datos: dict[str, Any], campo: str) -> float:
    valor = datos.get(campo)
    try:
        return float(valor)
    except (TypeError, ValueError):
        return 0.0


def indicadores_calculados(datos: dict[str, Any]) -> dict[str, float]:
    return {
        campo: valor_numerico(datos, campo)
        for campo in INDICADORES_OPERATIVOS
    }


def causa_principal_riesgo(datos: dict[str, Any], nivel_riesgo: str) -> str:
    if valor_numerico(datos, "ocupacion_estimada") >= 0.90:
        return "Ocupación crítica"
    if valor_numerico(datos, "ratio_camas_disponibles") <= 0.10:
        return "Capacidad disponible limitada"
    if valor_numerico(datos, "diferencia_ingresos_egresos") > 0:
        return "Demanda supera egresos"
    if valor_numerico(datos, "promedio_estancia") > 7:
        return "Estancia prolongada"
    if valor_numerico(datos, "presion_ingresos_camas") > 1:
        return "Alta presión ingresos/camas"
    if valor_numerico(datos, "rotacion_camas") < 1:
        return "Baja rotación de camas"
    if (nivel_riesgo or "").strip().lower() == "bajo":
        return "Riesgo controlado"
    return "Presión operativa multifactorial"


def calcular_brecha_operativa(
    datos: dict[str, Any],
    nivel_riesgo: str,
) -> dict[str, Any]:
    riesgo = (nivel_riesgo or "").strip().lower()
    ocupacion = valor_numerico(datos, "ocupacion_estimada")
    puntaje = 0

    if riesgo == "alto":
        puntaje += 30
    elif riesgo == "medio":
        puntaje += 15

    if ocupacion >= 0.90:
        puntaje += 25
    elif 0.80 <= ocupacion < 0.90:
        puntaje += 15

    if valor_numerico(datos, "presion_ingresos_camas") > 1:
        puntaje += 15
    if valor_numerico(datos, "diferencia_ingresos_egresos") > 0:
        puntaje += 10
    if valor_numerico(datos, "promedio_estancia") > 7:
        puntaje += 10
    if valor_numerico(datos, "rotacion_camas") < 1:
        puntaje += 5
    if valor_numerico(datos, "ratio_camas_disponibles") <= 0.10:
        puntaje += 10

    puntaje = min(puntaje, 100)
    if puntaje >= 70:
        nivel = "Brecha crítica"
    elif puntaje >= 40:
        nivel = "Brecha en observación"
    else:
        nivel = "Brecha controlada"

    return {
        "brecha_operativa": puntaje,
        "nivel_brecha_operativa": nivel,
    }


def diagnostico_operativo(nivel_riesgo: str, causa: str) -> str:
    riesgo = (nivel_riesgo or "sin datos").upper()
    return (
        "Para el siguiente mes, el servicio evaluado presenta riesgo "
        f"{riesgo} de insuficiencia de capacidad asistencial. La causa "
        f"principal identificada es {causa.lower()}. Esto indica que la "
        "demanda hospitalaria podría ejercer presión sobre la capacidad "
        "registrada si la tendencia continúa."
    )


def recomendaciones_operativas(causa: str) -> list[str]:
    recomendaciones = {
        "Ocupación crítica": [
            "Activar seguimiento operativo del servicio.",
            "Revisar disponibilidad registrada y posibles camas no operativas.",
            "Verificar que las camas liberadas sean reportadas oportunamente.",
            "Comunicar alerta a gestión hospitalaria y servicios involucrados.",
        ],
        "Demanda supera egresos": [
            "Revisar si los egresos programados compensan los ingresos esperados.",
            "Coordinar seguimiento de altas próximas.",
            "Revisar posibles demoras administrativas en egresos.",
            "Priorizar monitoreo del servicio.",
        ],
        "Estancia prolongada": [
            "Identificar pacientes con permanencia elevada.",
            "Revisar posibles demoras en exámenes, interconsultas, trámites o traslados.",
            "Coordinar seguimiento de pacientes con estancia prolongada.",
            "Evaluar impacto de la estancia en la rotación de camas.",
        ],
        "Capacidad disponible limitada": [
            "Verificar actualización de camas disponibles o habilitadas.",
            "Revisar si existen camas bloqueadas, en mantenimiento o no reportadas.",
            "Comunicar la limitación a gestión hospitalaria.",
        ],
        "Alta presión ingresos/camas": [
            "Revisar tendencia de ingresos del servicio.",
            "Coordinar seguimiento preventivo con hospitalización y admisión.",
            "Monitorear presión ingresos/camas durante el periodo siguiente.",
        ],
        "Baja rotación de camas": [
            "Revisar causas de baja rotación operativa.",
            "Coordinar seguimiento de egresos y estancias prolongadas.",
            "Monitorear disponibilidad registrada antes del siguiente mes.",
        ],
    }
    return recomendaciones.get(
        causa,
        [
            "Mantener monitoreo mensual del servicio.",
            "Revisar indicadores de demanda y capacidad antes del siguiente mes.",
            "Registrar acciones preventivas si el riesgo aumenta.",
        ],
    )


def acciones_prioritarias(causa: str, nivel_riesgo: str) -> list[str]:
    acciones = [
        "Revisar servicio prioritario.",
        "Revisar causa principal del riesgo.",
    ]
    if (nivel_riesgo or "").strip().lower() in {"medio", "alto"}:
        acciones.append("Comunicar alerta preventiva a gestión hospitalaria.")
    if causa != "Riesgo controlado":
        acciones.append("Registrar seguimiento de la recomendación operativa.")
    return acciones


def interpretacion_modelo(nivel_riesgo: str, confianza: float) -> str:
    return (
        "El modelo XGBoost clasifica el riesgo del siguiente mes en bajo, "
        f"medio o alto. Para este registro predice riesgo {nivel_riesgo.upper()} "
        f"con confianza {confianza:.1%}. El resultado es referencial y debe "
        "usarse como apoyo preventivo a la gestión hospitalaria."
    )


def generar_soporte_decision(
    datos: dict[str, Any],
    nivel_riesgo: str,
    confianza: float,
) -> dict[str, Any]:
    causa = causa_principal_riesgo(datos, nivel_riesgo)
    brecha = calcular_brecha_operativa(datos, nivel_riesgo)
    return {
        "indicadores_calculados": indicadores_calculados(datos),
        "causa_principal_riesgo": causa,
        **brecha,
        "diagnostico_operativo": diagnostico_operativo(nivel_riesgo, causa),
        "recomendaciones_operativas": recomendaciones_operativas(causa),
        "acciones_prioritarias": acciones_prioritarias(causa, nivel_riesgo),
        "interpretacion_modelo": interpretacion_modelo(nivel_riesgo, confianza),
        "confianza_prediccion": confianza,
    }
