# Plataforma predictiva de capacidad asistencial en IPRESS

Proyecto académico que clasifica el riesgo de insuficiencia de capacidad
asistencial en IPRESS públicas de Lima Metropolitana:

- `bajo` (`0`)
- `medio` (`1`)
- `alto` (`2`)

Trabaja con información mensual agregada por IPRESS, mes y servicio
hospitalario. No asigna camas automáticamente, no reemplaza decisiones clínicas
y no funciona como historia clínica electrónica.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datos multianuales

El proyecto lee todos los archivos `*.csv` disponibles en `data/raw/`, ordenados
por nombre. Ejemplos:

```text
ConsultaD1_Hospitalizaciones_Especialidad_2015_v1.csv
ConsultaD1_Hospitalizaciones_Especialidad_2016_v1.csv
ConsultaD1_Hospitalizaciones_Especialidad_2017_v1.csv
```

Se admiten CSV delimitados por coma o punto y coma. La columna histórica
`DIAS_CAMA_DISPONIBLE` se normaliza como `NRO_TOTAL_CAMAS_DISPONIB`. Cada
archivo debe contener las demás columnas obligatorias; si falta alguna, el
proceso informa el archivo y las columnas faltantes.

Los archivos vacíos se omiten con una advertencia. El pipeline no inventa
registros y falla si no existe ningún CSV válido.

```powershell
py src/preparar_dataset.py
```

El procesamiento:

1. Concatena todos los CSV y agrega `archivo_origen`.
2. Limpia y valida los registros.
3. Filtra IPRESS públicas de Lima Metropolitana.
4. Consolida por año, mes, IPRESS, servicio y archivo de origen.
5. Calcula indicadores y la variable objetivo.
6. Genera `data/processed/dataset_modelo_ipress.csv`.

## Indicadores

Las fórmulas están centralizadas en `src/indicadores.py` y son compartidas por
el procesamiento y la API:

- `dias_mes`: días reales del año y mes.
- `capacidad_mensual`: camas totales por días del mes.
- `promedio_estancia`: estancias / egresos.
- `tasa_fallecidos`: fallecidos / egresos.
- `ratio_camas_disponibles`: camas-día disponibles / capacidad mensual.
- `ocupacion_estimada`: pacientes-cama / capacidad mensual.
- `presion_ingresos_camas`: ingresos / camas totales.
- `rotacion_camas`: egresos / camas totales.
- `diferencia_ingresos_egresos`: ingresos - egresos.

## Entrenamiento y evaluación

```powershell
py src/entrenar_modelo.py
```

Se comparan Regresión Logística, Random Forest y XGBoost en dos modos:

- `completo`: excluye objetivo, nombre de IPRESS y archivo de origen.
- `interpretable`: también excluye `codigo_ipress`, `ubigeo` e
  `id_hospitalizacion` para reducir memorización.

Cada modo usa:

- Evaluación aleatoria estratificada con 20% de prueba.
- Evaluación temporal con el último año como prueba y los anteriores como
  entrenamiento, cuando hay más de un año.

El modo interpretable se selecciona para producción si logra F1 macro de al
menos `0.70` y no pierde más de `0.10` frente al modo completo. En caso
contrario se conserva el completo y se registra una advertencia.

Artefactos:

- `models/modelo_ipress.joblib`
- `models/metricas_modelo.csv`
- `models/metricas_modelo_aleatorio.csv`
- `models/metricas_modelo_temporal.csv`
- `models/metricas_modelo_completo.csv`
- `models/metricas_modelo_interpretable.csv`
- `models/clases_riesgo.json`
- `models/model_metadata.json`
- `models/importancia_variables.csv`
- `models/importancia_variables_completo.csv`
- `models/importancia_variables_interpretable.csv`

## API

```powershell
uvicorn src.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Documentación: <http://127.0.0.1:8000/docs>
- Salud: <http://127.0.0.1:8000/health>

`POST /predict` recibe los datos hospitalarios base y calcula los indicadores
internamente. La inferencia utiliza únicamente las columnas esperadas por el
modelo guardado, por lo que funciona tanto con el modo completo como con el
interpretable.

El resultado es referencial y no reemplaza decisiones clínicas ni asigna camas
automáticamente.

## Pruebas

```powershell
pytest
```

## Limitación metodológica

La variable objetivo se construye mediante reglas aplicadas a los indicadores
del mismo mes y esos indicadores también se usan como predictores. Por ello,
métricas muy altas miden principalmente la capacidad del modelo para reproducir
esas reglas; no demuestran por sí solas capacidad de pronóstico futuro. Para
validar capacidad predictiva real se requiere evaluación temporal y, de ser
posible, etiquetas observadas en periodos posteriores.

El proyecto procesa datos agregados y no requiere datos personales de pacientes.
