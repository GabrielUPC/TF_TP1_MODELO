from __future__ import annotations

from typing import Any


MENSAJE_REFERENCIAL = (
    "El resultado es referencial. No asigna camas automáticamente y no "
    "reemplaza decisiones clínicas."
)

_DICCIONARIO_PLANTILLA = {
    "codigo_ipress": "Código RENIPRESS o identificador de la IPRESS.",
    "anio": "Año del periodo mensual informado.",
    "mes": "Mes del periodo informado, con valores del 1 al 12.",
    "servicio_hospitalizacion": (
        "Servicio hospitalario analizado dentro de la IPRESS."
    ),
    "total_ingresos": "Cantidad de ingresos hospitalarios del mes.",
    "total_egresos": "Cantidad de egresos hospitalarios del mes.",
    "total_estancias": "Días acumulados de permanencia hospitalaria.",
    "total_pacientes_camas": "Uso acumulado de camas durante el mes.",
    "total_camas": "Capacidad física o registrada de camas.",
    "total_camas_disponibles": (
        "Capacidad mensual registrada como camas-día disponibles."
    ),
}

_INTERPRETACION_INDICADORES = {
    "ingresos_hospitalarios": "Demanda de entrada del mes.",
    "egresos_hospitalarios": (
        "Salida de pacientes y liberación de camas durante el mes."
    ),
    "estancias_hospitalarias": "Días acumulados de permanencia.",
    "pacientes_cama": "Uso acumulado de camas.",
    "camas_totales": "Capacidad física o registrada.",
    "capacidad_mensual_registrada": (
        "Camas-día disponibles del mes."
    ),
    "ocupacion_estimada": (
        "Pacientes-cama dividido entre la capacidad mensual registrada."
    ),
    "presion_ingresos_camas": (
        "Demanda mensual de ingresos frente a camas registradas."
    ),
}


def diccionario_plantilla() -> dict[str, str]:
    return dict(_DICCIONARIO_PLANTILLA)


def interpretacion_indicadores() -> dict[str, str]:
    return dict(_INTERPRETACION_INDICADORES)


def valor_numerico(datos: dict[str, Any], campo: str) -> float | None:
    valor = datos.get(campo)
    if valor is None:
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def porcentaje_desde_ratio(valor: float | None) -> float | None:
    if valor is None:
        return None
    return valor * 100


def formatear_ocupacion(valor: float | None) -> str:
    porcentaje = porcentaje_desde_ratio(valor)
    if porcentaje is None:
        return "Sin dato"
    return f"{porcentaje:.1f}%"


def clasificar_ocupacion(valor: float | None) -> str:
    if valor is None:
        return "sin dato"
    if valor >= 1.0:
        return "crítico / uso supera capacidad registrada"
    if valor >= 0.85:
        return "alto / presión alta"
    if valor >= 0.70:
        return "medio / presión moderada"
    return "bajo / estable"


def interpretar_semaforo(nivel_riesgo: str) -> dict[str, str]:
    riesgo = (nivel_riesgo or "").strip().lower()

    if riesgo == "alto":
        return {
            "color_semaforo": "rojo",
            "interpretacion_riesgo": (
                "Existe posible insuficiencia de capacidad asistencial para "
                "el siguiente mes."
            ),
            "recomendacion_riesgo": (
                "Se recomienda revisar camas habilitadas, validar registros "
                "procesados, analizar ocupación estimada y coordinar la "
                "revisión hospitalaria con el área responsable. "
                f"{MENSAJE_REFERENCIAL}"
            ),
        }

    if riesgo == "medio":
        return {
            "color_semaforo": "amarillo",
            "interpretacion_riesgo": (
                "Existen señales de presión hospitalaria que requieren "
                "revisión."
            ),
            "recomendacion_riesgo": (
                "Revisar indicadores, validar tendencia de ingresos y "
                "coordinar revisión hospitalaria. "
                f"{MENSAJE_REFERENCIAL}"
            ),
        }

    return {
        "color_semaforo": "verde",
        "interpretacion_riesgo": (
            "La capacidad parece estable frente a la demanda esperada."
        ),
        "recomendacion_riesgo": (
            "Mantener monitoreo mensual. "
            f"{MENSAJE_REFERENCIAL}"
        ),
    }


def factores_explicativos_riesgo(
    indicadores: dict[str, Any],
    nivel_riesgo: str,
) -> list[str]:
    ocupacion = valor_numerico(indicadores, "ocupacion_estimada")
    pacientes_cama = valor_numerico(indicadores, "total_pacientes_camas")
    capacidad = valor_numerico(indicadores, "total_camas_disponibles")
    estancias = valor_numerico(indicadores, "total_estancias")
    promedio_estancia = valor_numerico(indicadores, "promedio_estancia")
    presion = valor_numerico(indicadores, "presion_ingresos_camas")

    factores: list[str] = []
    riesgo = (nivel_riesgo or "").strip().lower()

    if ocupacion is not None:
        if ocupacion >= 1.0:
            factores.append(
                "El uso acumulado de camas supera la capacidad mensual "
                "registrada."
            )
        elif ocupacion >= 0.85:
            factores.append(
                "La ocupación estimada supera el umbral de presión alta."
            )
        elif ocupacion >= 0.70:
            factores.append(
                "La ocupación estimada muestra presión moderada."
            )
        else:
            factores.append(
                "La ocupación estimada se mantiene en un rango bajo o estable."
            )

    if (
        pacientes_cama is not None
        and capacidad is not None
        and capacidad > 0
        and pacientes_cama > capacidad
    ):
        factores.append(
            "Los pacientes-cama superan las camas-día disponibles del mes."
        )

    if (
        promedio_estancia is not None
        and promedio_estancia > 7
    ) or (
        estancias is not None
        and estancias > 0
        and riesgo in {"medio", "alto"}
    ):
        factores.append(
            "Las estancias hospitalarias reflejan uso prolongado de camas."
        )

    if presion is not None:
        if presion >= 1:
            factores.append(
                "La presión ingresos/camas indica alta demanda frente a la "
                "capacidad registrada."
            )
        elif riesgo == "medio":
            factores.append(
                "La presión ingresos/camas requiere revisión hospitalaria."
            )

    factores.append(
        "El modelo también considera comportamiento histórico, tendencias y "
        "características del servicio."
    )
    return factores
