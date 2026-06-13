# Integracion de Spring Boot con el modelo FastAPI

## Arquitectura oficial

La integracion del sistema sigue este flujo:

```text
Frontend Angular (4200)
        |
        v
Backend Spring Boot (8080)
        |
        v
Microservicio FastAPI (8000)
        |
        v
Modelo XGBoost
```

Angular no llama directamente a FastAPI. El frontend conserva
`environment.base = http://localhost:8080` y consume los endpoints del backend,
incluidos `/dashboard/resumen`, `/dashboard/detalle`, `/dashboard/alertas` y
`/dashboard/filtro`.

FastAPI incluye CORS para los origenes locales de Angular y Spring Boot como
apoyo durante el desarrollo. CORS no interviene en la llamada oficial
backend-a-backend realizada por Spring Boot.

## Levantar los servicios

### 1. FastAPI en el puerto 8000

Desde `TF_TP1_MODELO`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Comprobaciones:

```text
GET http://localhost:8000/health
GET http://localhost:8000/metadata
POST http://localhost:8000/predict
```

### 2. Spring Boot en el puerto 8080

Desde `TF_TP1_Backend`:

```powershell
.\mvnw.cmd spring-boot:run
```

El backend usa esta propiedad:

```properties
modelo.ipress.url=${MODELO_IPRESS_URL:http://localhost:8000}
```

El valor predeterminado sirve para desarrollo local. En otro entorno puede
cambiarse sin recompilar:

```powershell
$env:MODELO_IPRESS_URL="http://modelo-ipress:8000"
.\mvnw.cmd spring-boot:run
```

### 3. Angular en el puerto 4200

Desde `TF_TP1_Frontend`:

```powershell
npm install
npm start
```

## Flujo de una prediccion

1. El usuario carga el Excel desde Angular.
2. Spring Boot valida y guarda los registros hospitalarios.
3. El backend calcula `IndicadorHospitalario`.
4. Para cada indicador, `ModeloPredictivoClientService` envia un `POST` a
   `{modelo.ipress.url}/predict`.
5. El backend busca hasta los dos meses calendario anteriores de la misma
   IPRESS y el mismo servicio hospitalario.
6. FastAPI calcula indicadores y variables temporales, y ejecuta XGBoost.
7. Spring Boot guarda el resultado en `PrediccionRiesgo`.
8. El dashboard consulta exclusivamente los endpoints de Spring Boot.

Si FastAPI no responde, devuelve un estado invalido o entrega una respuesta
incompleta, Spring Boot informa un error `503` o `502`. No genera una prediccion
local silenciosa.

## Request enviado por Spring Boot

Ejemplo para marzo de 2026 con dos meses de historial:

```json
{
  "registro_actual": {
    "anio": 2026,
    "mes": 3,
    "ubigeo": "150101",
    "departamento": "LIMA",
    "provincia": "LIMA",
    "distrito": "LIMA",
    "sector": "MINSA",
    "categoria_ipress": "III-1",
    "codigo_ipress": "00006207",
    "id_hospitalizacion": "HOSPITALIZACION GENERAL",
    "servicio_hospitalizacion": "HOSPITALIZACION GENERAL",
    "total_ingresos": 80.0,
    "total_egresos": 70.0,
    "total_estancias": 350.0,
    "total_pacientes_camas": 2500.0,
    "total_camas": 100.0,
    "total_camas_disponibles": 90.0,
    "total_fallecidos": 0.0
  },
  "historial_ultimos_meses": [
    {
      "anio": 2026,
      "mes": 1,
      "ubigeo": "150101",
      "departamento": "LIMA",
      "provincia": "LIMA",
      "distrito": "LIMA",
      "sector": "MINSA",
      "categoria_ipress": "III-1",
      "codigo_ipress": "00006207",
      "id_hospitalizacion": "HOSPITALIZACION GENERAL",
      "servicio_hospitalizacion": "HOSPITALIZACION GENERAL",
      "total_ingresos": 72.0,
      "total_egresos": 68.0,
      "total_estancias": 330.0,
      "total_pacientes_camas": 2400.0,
      "total_camas": 100.0,
      "total_camas_disponibles": 92.0,
      "total_fallecidos": 0.0
    },
    {
      "anio": 2026,
      "mes": 2,
      "ubigeo": "150101",
      "departamento": "LIMA",
      "provincia": "LIMA",
      "distrito": "LIMA",
      "sector": "MINSA",
      "categoria_ipress": "III-1",
      "codigo_ipress": "00006207",
      "id_hospitalizacion": "HOSPITALIZACION GENERAL",
      "servicio_hospitalizacion": "HOSPITALIZACION GENERAL",
      "total_ingresos": 76.0,
      "total_egresos": 69.0,
      "total_estancias": 340.0,
      "total_pacientes_camas": 2450.0,
      "total_camas": 100.0,
      "total_camas_disponibles": 91.0,
      "total_fallecidos": 0.0
    }
  ]
}
```

Si falta uno o ambos meses anteriores, el backend envia una lista parcial o
vacia. FastAPI procesa la solicitud y completa `advertencia_historial`.

## Response de FastAPI

```json
{
  "periodo_actual": "2026-03",
  "periodo_predicho": "2026-04",
  "horizonte_prediccion": "mes_siguiente",
  "nivel_riesgo_predicho": "alto",
  "nivel_riesgo_codificado": 2,
  "probabilidad": 0.87,
  "probabilidades_por_clase": {
    "bajo": 0.03,
    "medio": 0.10,
    "alto": 0.87
  },
  "variables_principales": [
    {
      "variable": "ocupacion_estimada",
      "valor": 0.81
    }
  ],
  "advertencia_historial": null,
  "mensaje": "El resultado es referencial y no reemplaza decisiones clinicas ni asigna camas automaticamente."
}
```

## Persistencia en PrediccionRiesgo

Spring Boot actualiza o crea la prediccion asociada al indicador:

| Campo | Valor |
|---|---|
| `nivelRiesgo` | `nivel_riesgo_predicho` convertido a `BAJO`, `MEDIO` o `ALTO` |
| `probabilidad` | `probabilidad` |
| `modeloUtilizado` | `XGBoost - FastAPI` |
| `fechaPrediccion` | `LocalDateTime.now()` |

## Mapeo y valores predeterminados

| FastAPI | Spring Boot |
|---|---|
| `codigo_ipress` | `Ipress.codigoRenipress` |
| `categoria_ipress` | `Ipress.categoriaIpress` |
| `ubigeo` | `Ipress.codigoUbigeo` |
| `distrito` | `Ipress.distrito` |
| `provincia` | `Ipress.provincia` |
| `departamento` | `Ipress.departamento` |
| `servicio_hospitalizacion` | `RegistroHospitalario.servicioHospitalario` |
| `total_ingresos` | `RegistroHospitalario.ingresos` |
| `total_egresos` | `RegistroHospitalario.egresos` |
| `total_estancias` | `RegistroHospitalario.estancias` |
| `total_pacientes_camas` | `RegistroHospitalario.pacientesCama` |
| `total_camas` | `RegistroHospitalario.camasTotales` |
| `total_camas_disponibles` | `RegistroHospitalario.camasDisponiblesHabilitadas` |

Valores predeterminados importantes para la sustentacion:

- `sector`: `"MINSA"`, porque la entidad actual no almacena el sector.
- `id_hospitalizacion`: `servicioHospitalario`, porque no existe un codigo
  especifico de hospitalizacion en el backend.
- `total_fallecidos`: `0.0`, porque el Excel y la entidad actual no lo registran.

Los datos geograficos, categoria y codigo RENIPRESS son obligatorios. Si la
IPRESS no los tiene, el backend detiene la prediccion con un error claro en vez
de enviar valores inventados al modelo.

## Campos mostrados en Angular

No se cambia el contrato de `DashboardDetalleDTO`. El frontend sigue recibiendo:

- `nivelRiesgo`
- `probabilidad`
- `modeloUtilizado`
- `fechaPrediccion`
- `ocupacionEstimada`
- `presionIngresosCamas`
- `promedioEstancia`
- `rotacionCamas`

Los endpoints del dashboard mantienen sus rutas y respuestas actuales.

## Validacion manual

1. Levantar FastAPI en `localhost:8000`.
2. Confirmar `GET /health`.
3. Levantar Spring Boot en `localhost:8080`.
4. Levantar Angular en `localhost:4200`.
5. Iniciar sesion y cargar un Excel valido.
6. Verificar que `PrediccionRiesgo.modeloUtilizado` sea
   `XGBoost - FastAPI`.
7. Abrir el dashboard y comprobar `nivelRiesgo` y `probabilidad`.
