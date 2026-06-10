# Plataforma predictiva de capacidad asistencial en IPRESS

Proyecto académico para clasificar el riesgo de insuficiencia de capacidad
asistencial en IPRESS públicas de Lima Metropolitana usando información mensual
agregada por establecimiento y servicio hospitalario.

El modelo analiza disponibilidad y uso de camas. No asigna camas
automáticamente, no reemplaza decisiones clínicas y no funciona como historia
clínica electrónica.

## Estructura

```text
TF_TP1_Modelo/
|-- data/
|   |-- raw/
|   |   `-- ConsultaD1_Hospitalizaciones_Especialidad_2015_v1.csv
|   `-- processed/
|       `-- dataset_modelo_ipress.csv
|-- models/
|   |-- modelo_ipress.joblib
|   |-- metricas_modelo.csv
|   `-- clases_riesgo.json
|-- src/
|   |-- preparar_dataset.py
|   |-- entrenar_modelo.py
|   |-- predecir.py
|   `-- main.py
|-- requirements.txt
`-- README.md
```

## Instalación

Desde la raíz del proyecto, crear el entorno virtual:

```powershell
python -m venv .venv
```

Activarlo en PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
pip install -r requirements.txt
```

## Preparación y entrenamiento

Preparar y consolidar el dataset:

```powershell
py src/preparar_dataset.py
```

El proceso filtra IPRESS públicas de Lima Metropolitana, calcula indicadores
hospitalarios y genera `data/processed/dataset_modelo_ipress.csv`.

Entrenar y comparar los modelos:

```powershell
py src/entrenar_modelo.py
```

Se entrenan Regresión Logística, Random Forest y XGBoost. Si XGBoost no está
instalado o no puede importarse, el proceso continúa con los otros dos modelos.
El mejor modelo según F1 macro se guarda en `models/modelo_ipress.joblib`.

## API

Ejecutar la API:

```powershell
uvicorn src.main:app --reload
```

Direcciones:

- API: <http://127.0.0.1:8000>
- Documentación interactiva: <http://127.0.0.1:8000/docs>
- Estado: <http://127.0.0.1:8000/health>

El endpoint `POST /predict` recibe una observación mensual agregada y devuelve
una clasificación:

- `bajo` (`0`)
- `medio` (`1`)
- `alto` (`2`)

La probabilidad y la clasificación son resultados referenciales. No reemplazan
decisiones clínicas ni asignan camas automáticamente.

## Limitación metodológica

La variable objetivo se construye mediante reglas aplicadas a los indicadores
del mismo mes y esos indicadores también se usan como predictores. Por ello, una
métrica de prueba muy alta mide principalmente la capacidad del modelo para
reproducir esas reglas; no demuestra por sí sola capacidad de pronóstico futuro.
Una validación prospectiva requeriría datos de varios periodos y una etiqueta de
riesgo observada en meses posteriores.

## Alcance de los datos

El flujo usa datos estructurados y agregados por IPRESS, mes y servicio
hospitalario. No requiere ni procesa datos privados identificables de pacientes.
