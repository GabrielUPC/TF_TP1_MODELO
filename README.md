# Plataforma predictiva de capacidad asistencial en IPRESS

Proyecto académico que clasifica el riesgo de insuficiencia de capacidad
asistencial en IPRESS públicas de Lima Metropolitana:

- `bajo` (`0`)
- `medio` (`1`)
- `alto` (`2`)

Trabaja con información mensual agregada por IPRESS, mes y servicio
hospitalario. No asigna camas automáticamente, no reemplaza decisiones clínicas
y no funciona como historia clínica electrónica.

## Estructura

```text
TF_TP1_Modelo/
|-- data/
|   |-- raw/
|   |   |-- README.md
|   |   `-- ConsultaD1_Hospitalizaciones_Especialidad_2015_v1.csv
|   `-- processed/
|       `-- dataset_modelo_ipress.csv
|-- models/
|   |-- modelo_ipress.joblib
|   |-- metricas_modelo.csv
|   |-- clases_riesgo.json
|   |-- model_metadata.json
|   `-- importancia_variables.csv
|-- src/
|   |-- indicadores.py
|   |-- preparar_dataset.py
|   |-- entrenar_modelo.py
|   |-- predecir.py
|   `-- main.py
|-- tests/
|-- requirements.txt
`-- README.md
```

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

El CSV original puede no distribuirse con el repositorio. Antes del
procesamiento debe colocarse en:

```text
data/raw/ConsultaD1_Hospitalizaciones_Especialidad_2015_v1.csv
```

El pipeline no inventa registros. Informa claramente si el archivo no existe,
está vacío, no contiene filas o carece de columnas obligatorias.

Para preparar el dataset:

```powershell
py src/preparar_dataset.py
```

El proceso limpia, consolida y filtra IPRESS públicas de Lima Metropolitana.
El resultado queda en `data/processed/dataset_modelo_ipress.csv`.

## Indicadores

Las fórmulas están centralizadas en `src/indicadores.py` y son compartidas por
el procesamiento y la API:

- `dias_mes`: días reales del año y mes.
- `capacidad_mensual`: camas totales por días del mes.
- `promedio_estancia`: estancias totales / egresos.
- `tasa_fallecidos`: fallecidos / egresos.
- `ratio_camas_disponibles`: camas-día disponibles / capacidad mensual.
- `ocupacion_estimada`: pacientes-cama / capacidad mensual.
- `presion_ingresos_camas`: ingresos / camas totales.
- `rotacion_camas`: egresos / camas totales.
- `diferencia_ingresos_egresos`: ingresos - egresos.

Las divisiones entre cero se controlan de forma explícita.

## Entrenamiento

```powershell
py src/entrenar_modelo.py
```

Se comparan Regresión Logística, Random Forest y XGBoost, cuando está
disponible. El mejor modelo se selecciona por F1 macro y genera:

- `models/modelo_ipress.joblib`
- `models/metricas_modelo.csv`
- `models/clases_riesgo.json`
- `models/model_metadata.json`
- `models/importancia_variables.csv`, si el modelo expone importancias

## API

```powershell
uvicorn src.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Documentación: <http://127.0.0.1:8000/docs>
- Salud: <http://127.0.0.1:8000/health>

`POST /predict` recibe únicamente datos base hospitalarios. La API calcula los
indicadores internamente antes de invocar el modelo, por lo que el consumidor no
debe duplicar esas fórmulas.

El resultado incluye la clase, probabilidades y este mensaje:

> El resultado es referencial y no reemplaza decisiones clínicas ni asigna
> camas automáticamente.

## Pruebas

```powershell
pytest
```

Las pruebas cubren indicadores, validación del CSV, salud de la API, cálculo
interno de indicadores y ausencia del modelo.

## Limitación metodológica

La variable objetivo se construye mediante reglas aplicadas a los indicadores
del mismo mes y esos indicadores también se usan como predictores. Por ello, una
métrica de prueba muy alta mide principalmente la capacidad del modelo para
reproducir esas reglas; no demuestra por sí sola capacidad de pronóstico futuro.
Una validación prospectiva requeriría datos de varios periodos y una etiqueta de
riesgo observada en meses posteriores.

El proyecto procesa datos agregados y no requiere datos personales de pacientes.
