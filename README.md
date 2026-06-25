# Plataforma predictiva de capacidad asistencial en IPRESS

Microservicio de machine learning que predice el riesgo de insuficiencia de
capacidad asistencial del **siguiente mes** en IPRESS públicas de Lima
Metropolitana.

El sistema trabaja con información mensual agregada por IPRESS y servicio
hospitalario. Clasifica el riesgo como:

- `bajo` (`0`)
- `medio` (`1`)
- `alto` (`2`)

El resultado es referencial. No reemplaza decisiones clínicas, no asigna camas
automáticamente, no decide hospitalizaciones, no funciona como historia clínica
electrónica y no sustituye los sistemas internos de una IPRESS.

## Alcance, granularidad, indicadores y control de carga

El alcance actual usa la Tabla D1 de hospitalización y camas. No incorpora
todavía tablas A, J ni H; esas fuentes quedan como ampliaciones futuras para
enriquecer contexto, no para cambiar el contrato principal ya integrado.

La unidad de análisis es mensual:

```text
1 registro = 1 IPRESS + 1 mes + 1 servicio hospitalario
```

La plataforma no trabaja a nivel de paciente, no interpreta eventos diarios, no
opera en tiempo real, no asigna camas automáticamente y no reemplaza decisiones
clínicas. La predicción usa los datos del mes actual para estimar el riesgo del
siguiente mes.

La carga mensual se valida con:

```powershell
py src\validar_plantilla.py ruta_del_archivo.csv
```

También acepta archivos Excel (`.xlsx` o `.xls`). El validador normaliza alias
como `codigo_renipress`, `servicio_hospitalario`, `ingresos`, `egresos`,
`estancias`, `pacientes_cama`, `camas_totales` y
`camas_disponibles_habilitadas` hacia las columnas canónicas del modelo.

La clave de granularidad vigente es:

```text
codigo_ipress + anio + mes + servicio_hospitalizacion
```

No se permite más de un registro vigente con la misma IPRESS, año, mes y
servicio hospitalario.

Los indicadores se interpretan como señales de presión operativa. Por ejemplo,
`ocupacion_estimada` es un ratio y debe mostrarse como porcentaje: `0.85` se
lee como `85%`, `1.37` como `137%` y `1.4` como `140%`. Un valor mayor o igual
a `100%` indica que el uso acumulado supera la capacidad mensual registrada.

El riesgo predicho se comunica como semáforo operacional:

- `bajo`: capacidad aparentemente estable frente a la demanda esperada.
- `medio`: señales de presión hospitalaria que requieren seguimiento.
- `alto`: posible insuficiencia de capacidad asistencial para el siguiente mes.

La respuesta incluye recomendación y factores explicativos. Estos factores
resumen presiones como ocupación alta, uso que supera capacidad, estancias
prolongadas, presión ingresos/camas y el comportamiento histórico, tendencias y
características del servicio consideradas por el modelo.

## Soporte operativo a la decisión

Además de predecir riesgo `bajo`, `medio` o `alto` del siguiente mes, `/predict`
devuelve soporte operativo para que el usuario comprenda el resultado.
Este soporte se genera con reglas transparentes sobre indicadores mensuales de
demanda y capacidad, sin cambiar el modelo XGBoost ni el horizonte predictivo.

La respuesta incluye:

- `indicadores_calculados`: ocupación estimada, presión ingresos/camas,
  promedio de estancia, rotación, diferencia ingresos-egresos, ratio de camas
  disponibles y capacidad registrada.
- `causa_principal_riesgo`: causa operativa dominante, por ejemplo ocupación
  crítica, demanda supera egresos, estancia prolongada o capacidad disponible
  limitada.
- `brecha_operativa` y `nivel_brecha_operativa`: puntaje preventivo de 0 a 100
  clasificado como brecha controlada, en observación o crítica.
- `diagnostico_operativo`: lectura del riesgo del siguiente mes en lenguaje de
  gestión hospitalaria.
- `recomendaciones_operativas`: recomendaciones generales de apoyo para revisar
  indicadores de demanda y capacidad.
- `interpretacion_modelo`, `confianza_prediccion` y probabilidades explícitas
  por clase.

La brecha operativa no representa una falta exacta de camas en tiempo real. Es
una señal preventiva para interpretar la presión entre demanda y capacidad
registrada. El sistema no crea camas, no asigna camas, no decide altas y no
reemplaza decisiones clínicas.

## Flujo predictivo

El rediseño cambia el problema anterior:

```text
indicadores del mes t -> riesgo del mismo mes t
```

por un horizonte futuro:

```text
datos e indicadores del mes t -> riesgo del mes t+1
```

Ejemplos:

- Enero de 2024 predice febrero de 2024.
- Febrero de 2024 predice marzo de 2024.
- Diciembre de 2024 predice enero de 2025.

Solo se construye una pareja cuando existe exactamente el siguiente mes
calendario para la misma IPRESS y servicio. No se enlazan periodos con vacíos.

## Arquitectura

```text
data/raw/*.csv
       |
       v
src/preparar_dataset.py
       |
       v
data/processed/dataset_modelo_ipress.csv
       |
       v
src/entrenar_modelo.py
       |
       v
models/modelo_ipress.joblib
       |
       v
FastAPI: /predict, /metadata, /health
```

El modelo queda preparado como microservicio para integrarse posteriormente con
backend, frontend, dashboard y despliegue en nube. La arquitectura permite
incorporar en el futuro variables climáticas, epidemiológicas u otras fuentes
contextuales sin cambiar el contrato central.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Preparación de datos

Coloque los CSV en `data/raw/` y ejecute:

```powershell
py src/preparar_dataset.py
```

El pipeline:

1. Lee todos los CSV multianuales.
2. Detecta coma o punto y coma y admite `utf-8-sig` o `latin1`.
3. Omite archivos vacíos o temporales con una advertencia.
4. Normaliza nombres y el alias
   `DIAS_CAMA_DISPONIBLE -> NRO_TOTAL_CAMAS_DISPONIB`.
5. Limpia duplicados, textos y valores numéricos inválidos.
6. Filtra IPRESS públicas de Lima Metropolitana.
7. Consolida por IPRESS, servicio y periodo mensual.
8. Calcula indicadores hospitalarios.
9. Calcula variables estacionales, móviles y de tendencia.
10. Construye el riesgo actual mediante reglas de negocio.
11. Desplaza ese riesgo para crear el objetivo del siguiente mes.
12. Elimina filas sin un mes siguiente continuo.

Salidas:

- `data/processed/dataset_modelo_ipress.csv`
- `data/processed/dataset_metadata.json`

### Riesgo actual

El riesgo actual se utiliza únicamente para construir la etiqueta futura y los
baselines. No es una variable predictora del modelo.

Riesgo alto:

- `ocupacion_estimada >= 0.85`, o
- `ratio_camas_disponibles <= 0.10`, o
- `presion_ingresos_camas >= percentil 75`.

Riesgo medio:

- `ocupacion_estimada >= 0.70`, o
- `ratio_camas_disponibles <= 0.20`, o
- `presion_ingresos_camas >= percentil 50`.

Los percentiles se calculan sobre valores finitos después de limpieza, filtro y
consolidación. En la ejecución actual fueron `P50 = 2.60` y `P75 = 4.85`.

### Variables temporales

Se agregan trimestre, semestre, seno y coseno del mes, fin de año, promedios
móviles de tres meses y tendencias de un mes.

Los promedios móviles usan el mes actual y hasta dos meses anteriores. Las
tendencias comparan el mes actual con el anterior. Si hay un vacío mensual, la
tendencia se neutraliza y no se arrastra información de un periodo distante.
Ninguna variable usa meses futuros.

## Indicadores

Las fórmulas están centralizadas en `src/indicadores.py`:

- `dias_mes`: días reales del año y mes.
- `capacidad_mensual`: camas totales por días del mes.
- `promedio_estancia`: estancias / egresos.
- `tasa_fallecidos`: fallecidos / egresos.
- `ratio_camas_disponibles`: camas-día disponibles / capacidad mensual.
- `ocupacion_estimada`: pacientes-cama / capacidad mensual.
- `presion_ingresos_camas`: ingresos / camas totales.
- `rotacion_camas`: egresos / camas totales.
- `diferencia_ingresos_egresos`: ingresos - egresos.

Las divisiones entre cero están controladas. Si hay ingresos con cero camas, la
presión conserva el número de ingresos como señal extrema y el procesamiento
registra una advertencia.

## Entrenamiento

```powershell
py src/entrenar_modelo.py
```

Para elegir manualmente el año de prueba:

```powershell
py src/entrenar_modelo.py --anio-prueba 2025
```

Se comparan:

- Regresión Logística
- Random Forest
- XGBoost

Existe un único flujo oficial de variables. El algoritmo ganador se selecciona
por **F1 macro temporal**. La evaluación aleatoria se conserva solamente como
referencia secundaria.

Por defecto se usa como prueba el último año objetivo con los doce meses
disponibles. Como 2026 contiene enero a abril, la ejecución actual usa **2025**
como prueba y años anteriores como entrenamiento.

### Desbalance

La distribución futura actual es:

- bajo: 8,187
- medio: 10,311
- alto: 27,811

Se usa `class_weight="balanced"` en Regresión Logística y Random Forest. Para
XGBoost se calculan pesos balanceados solo con el conjunto de entrenamiento. No
se usa SMOTE y nunca se modifica el conjunto de prueba.

### Resultados actuales

XGBoost fue seleccionado como modelo oficial.

| Evaluación | Accuracy | F1 macro | ROC-AUC OVR macro |
|---|---:|---:|---:|
| Temporal 2025 | 0.7904 | 0.7546 | 0.9236 |
| Aleatoria de referencia | 0.7547 | 0.7164 | 0.8989 |

Baselines temporales:

| Baseline | F1 macro |
|---|---:|
| Clase mayoritaria | 0.2549 |
| Riesgo siguiente igual al actual | 0.7303 |
| Regla basada solo en ocupación | 0.5615 |

El modelo supera al mejor baseline por aproximadamente `0.0243` de F1 macro. El
margen mínimo configurado es `0.02`, por lo que `supera_baseline = true`, aunque
la diferencia debe interpretarse con prudencia.

Los principales grupos de variables fueron servicio hospitalario, distrito,
ocupación estimada, presión de ingresos por cama, categoría IPRESS, promedio
móvil de presión y promedio móvil de ocupación.

### Artefactos

- `models/modelo_ipress.joblib`
- `models/model_metadata.json`
- `models/clases_riesgo.json`
- `models/metricas_modelo.csv`
- `models/metricas_modelo_temporal.csv`
- `models/metricas_modelo_aleatorio.csv`
- `models/metricas_baseline.csv`
- `models/matriz_confusion_temporal.csv`
- `models/matriz_confusion_aleatoria.csv`
- `models/classification_report_temporal.json`
- `models/classification_report_aleatorio.json`
- `models/importancia_variables.csv`

## API

```powershell
uvicorn src.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Swagger: <http://127.0.0.1:8000/docs>
- Salud: <http://127.0.0.1:8000/health>
- Metadata: <http://127.0.0.1:8000/metadata>

### POST /predict

La API recibe los datos base del mes actual y hasta doce registros históricos
opcionales. Para los promedios de tres meses son suficientes los dos meses
anteriores.

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
    "id_hospitalizacion": "241500",
    "servicio_hospitalizacion": "HOSPITALIZACION GENERAL",
    "total_ingresos": 80,
    "total_egresos": 70,
    "total_estancias": 350,
    "total_pacientes_camas": 2500,
    "total_camas": 100,
    "total_camas_disponibles": 3100,
    "total_fallecidos": 2
  },
  "historial_ultimos_meses": []
}
```

La API calcula internamente indicadores, estacionalidad, promedios móviles y
tendencias. Si no recibe los dos meses previos, utiliza la información
disponible y devuelve `advertencia_historial`.

La respuesta incluye periodo actual, periodo predicho, riesgo, semáforo,
probabilidad por clase, variables principales, factores explicativos,
recomendación, advertencia de historial y mensaje de alcance.

Para revisar métricas sin reentrenar:

```powershell
py src\ver_metricas.py
```

### GET /metadata

Expone trazabilidad no sensible: tipo y horizonte del modelo, algoritmo,
fecha de entrenamiento, año de prueba, F1 temporal, variables, estrategia de
desbalance, comparación con baselines y advertencia metodológica.

El frontend podrá mostrar riesgo, probabilidad, periodo actual, periodo
predicho, variables principales y trazabilidad sin acceder a archivos internos.

## Pruebas

```powershell
pytest
```

Las pruebas cubren indicadores, lectura de CSV, objetivo futuro, continuidad
mensual, ausencia de fuga, variables móviles, baselines, API y metadata.

## Docker

El artefacto `models/modelo_ipress.joblib` se versiona y se copia dentro de la
imagen. Los datos raw, el entorno virtual y las cachés quedan excluidos.

```powershell
docker build -t modelo-ipress .
docker run -p 8000:8000 modelo-ipress
```

La imagen ejecuta:

```text
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Justificación metodológica del rediseño

El modelo fue rediseñado para predecir el riesgo de insuficiencia de capacidad
asistencial del siguiente periodo mensual. Esta decisión evita que el modelo
clasifique el riesgo del mismo mes usando los mismos indicadores con los que fue
construida la etiqueta, reduciendo el riesgo de fuga de información.

El enfoque se inspira en estudios de machine learning aplicados a regulación y
planificación hospitalaria, donde se recomienda limpiar rigurosamente los datos,
evaluar múltiples métricas, considerar el desbalance de clases, comparar
modelos y diseñar plataformas predictivas integrables con sistemas de
información.

De Barreto et al. (2024) se incorporan la limpieza rigurosa, comparación de
modelos, matrices de confusión, métricas múltiples, control de desbalance y
trazabilidad. De Cabral-Miranda et al. (2025) se adopta el enfoque de plataforma,
el horizonte futuro, la evaluación temporal, la API y la preparación para
despliegue escalable.

La etiqueta futura sigue construida mediante reglas de negocio, por lo que el
resultado representa una predicción operacional y no una validación clínica
independiente. Sirve como apoyo a la gestión hospitalaria, sin reemplazar
decisiones clínicas ni asignar camas automáticamente.

## Explicación breve para sustentación

El modelo anterior obtenía métricas casi perfectas porque aprendía una etiqueta
del mismo mes construida con sus propias variables de entrada. El rediseño usa
el mes actual para anticipar el siguiente, incorpora únicamente historia
disponible hasta el momento de predicción y evalúa sobre un año futuro completo.

La defensa principal no es solo el valor de accuracy: se presentan F1 macro,
balanced accuracy, especificidad, ROC-AUC, matriz de confusión, reportes por
clase y tres baselines. El modelo mejora la persistencia mensual, aunque por un
margen moderado, lo cual ofrece una interpretación más prudente y defendible
que las métricas casi perfectas del enfoque anterior.
