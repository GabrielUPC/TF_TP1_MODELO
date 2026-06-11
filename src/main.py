from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.indicadores import agregar_indicadores_registro
from src.predecir import predecir_riesgo


MENSAJE_REFERENCIAL = (
    "El resultado es referencial y no reemplaza decisiones clínicas ni asigna "
    "camas automáticamente."
)

app = FastAPI(
    title="API del modelo predictivo IPRESS",
    description=(
        "Clasifica el riesgo de insuficiencia de capacidad asistencial a partir "
        "de información hospitalaria mensual agregada."
    ),
    version="1.0.0",
)


class DatosPrediccion(BaseModel):
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


class ResultadoPrediccion(BaseModel):
    nivel_riesgo: str
    nivel_riesgo_codificado: int
    probabilidad: float
    probabilidades_por_clase: dict[str, float]
    mensaje: str


@app.get("/")
def raiz() -> dict[str, str]:
    return {
        "mensaje": "API del modelo predictivo IPRESS funcionando",
        "proyecto": (
            "Predicción de riesgo de insuficiencia de capacidad asistencial"
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=ResultadoPrediccion)
def predict(datos: DatosPrediccion) -> dict:
    try:
        registro_completo = agregar_indicadores_registro(datos.model_dump())
        resultado = predecir_riesgo(registro_completo)
        resultado["mensaje"] = MENSAJE_REFERENCIAL
        return resultado
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
