from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.indicadores import agregar_indicadores_registro
from src.predecir import obtener_metadata_publica, predecir_riesgo
from src.variables_temporales import (
    periodo_siguiente,
    preparar_registro_con_historial,
)


MENSAJE_REFERENCIAL = (
    "El resultado es referencial y no reemplaza decisiones clínicas ni asigna "
    "camas automáticamente."
)
ADVERTENCIA_HISTORIAL_INCOMPLETO = (
    "No se recibió historial completo de los dos meses previos; algunas "
    "variables temporales fueron estimadas con información limitada."
)

app = FastAPI(
    title="API predictiva de capacidad asistencial IPRESS",
    description=(
        "Predice el riesgo de insuficiencia de capacidad asistencial del "
        "siguiente mes usando información hospitalaria mensual agregada."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:8080",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DatosHospitalarios(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anio: int = Field(ge=1900, le=2100)
    mes: int = Field(ge=1, le=12)
    ubigeo: str = Field(min_length=1)
    departamento: str = Field(min_length=1)
    provincia: str = Field(min_length=1)
    distrito: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    categoria_ipress: str = Field(min_length=1)
    codigo_ipress: str = Field(min_length=1)
    id_hospitalizacion: str = Field(min_length=1)
    servicio_hospitalizacion: str = Field(min_length=1)
    total_ingresos: float = Field(ge=0)
    total_egresos: float = Field(ge=0)
    total_estancias: float = Field(ge=0)
    total_pacientes_camas: float = Field(ge=0)
    total_camas: float = Field(ge=0)
    total_camas_disponibles: float = Field(ge=0)
    total_fallecidos: float = Field(ge=0)


class SolicitudPrediccion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registro_actual: DatosHospitalarios
    historial_ultimos_meses: list[DatosHospitalarios] = Field(
        default_factory=list,
        max_length=12,
    )


class VariablePrincipal(BaseModel):
    variable: str
    valor: Any


class ResultadoPrediccion(BaseModel):
    periodo_actual: str
    periodo_predicho: str
    horizonte_prediccion: str
    nivel_riesgo_predicho: str
    nivel_riesgo_codificado: int
    probabilidad: float
    riesgo_insuficiencia_capacidad: float
    probabilidades_por_clase: dict[str, float]
    variables_principales: list[VariablePrincipal]
    advertencia_historial: str | None
    mensaje: str


@app.get("/")
def raiz() -> dict[str, str]:
    return {
        "mensaje": "API predictiva IPRESS funcionando",
        "proyecto": (
            "Predicción del riesgo de insuficiencia de capacidad asistencial "
            "del siguiente mes"
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    try:
        return obtener_metadata_publica()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/predict", response_model=ResultadoPrediccion)
def predict(solicitud: SolicitudPrediccion) -> dict[str, Any]:
    try:
        actual = agregar_indicadores_registro(
            solicitud.registro_actual.model_dump()
        )
        historial = [
            agregar_indicadores_registro(registro.model_dump())
            for registro in solicitud.historial_ultimos_meses
        ]
        fila_actual, historial_completo = preparar_registro_con_historial(
            actual,
            historial,
        )
        datos_modelo = fila_actual.iloc[0].to_dict()
        resultado = predecir_riesgo(datos_modelo)
        periodo_actual = (
            f"{solicitud.registro_actual.anio:04d}-"
            f"{solicitud.registro_actual.mes:02d}"
        )
        resultado.update(
            {
                "periodo_actual": periodo_actual,
                "periodo_predicho": periodo_siguiente(
                    solicitud.registro_actual.anio,
                    solicitud.registro_actual.mes,
                ),
                "horizonte_prediccion": "mes_siguiente",
                "advertencia_historial": (
                    None
                    if historial_completo
                    else ADVERTENCIA_HISTORIAL_INCOMPLETO
                ),
                "mensaje": MENSAJE_REFERENCIAL,
            }
        )
        return resultado
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
