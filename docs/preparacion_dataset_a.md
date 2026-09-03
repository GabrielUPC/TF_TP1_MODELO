# Preparación analítica de Dataset A

Versión: `dataset_a_analitico_v1`. Ejecución independiente:

```powershell
python -m pytest -q
python -m src.preparar_dataset_a
```

Esta fase aplica el tratamiento autorizado después de la auditoría exploratoria
de `docs/auditoria_dataset_a.md`. No modifica esa auditoría ni sus resultados
históricos. No entrena, no cambia features del modelo vigente y no sustituye
`data/processed/dataset_modelo_ipress.csv` ni archivos `.joblib`.

## Fuentes y granularidad

Se leen recursivamente los ConsultaA dentro de `data/raw` reutilizando el lector
y normalizador de la auditoría (texto, coma/punto y coma, UTF-8/Latin-1).
Actualmente están en `data/raw/Capacidad`. D procede exclusivamente del dataset
procesado existente, no se regenera desde RAW. Si falta A, faltan columnas
requeridas o hay un archivo ilegible, el proceso falla sin omitir la fuente.

A analítico conserva **todas las claves nacionales no ambiguas** de A;
no se limita a la vista de Lima del informe exploratorio. El cruce conserva
el alcance de D ya filtrado: IPRESS, servicio y mes observado `t`.

Se normaliza IPRESS a ocho dígitos sin truncar, con espacios exteriores y sufijo
`.0` permitidos; año y mes deben ser enteros válidos. La clave A es
`codigo_ipress + anio + mes`. Las claves inválidas se excluyen y registran con
su representación original. Los valores originales D se preservan: únicamente
las claves auxiliares del join se normalizan en memoria.

## Duplicados y trazabilidad

La comparación usa **todas las columnas originales como texto**, incluidas las
que no serán covariables, antes de normalizar cantidades. Archivo y número de
registro no intervienen en la igualdad. No basta que coincidan los cuatro recursos,
ni se decide la igualdad por una huella hash. Diferencias de formato de valores
se consideran conservadoramente diferencias de contenido, no se adivinan equivalencias.

- `UNICA`: una fila para esa clave.
- `DUPLICADO_EXACTO`: todas las filas coinciden; se conserva una sola, sin sumar.
  Se registran filas originales y cantidad eliminada en un resumen por clave.
- `AMBIGUA`: al menos dos versiones difieren. Se excluye la clave completa de A
  utilizable, sin sumar, promediar, escoger máximos/mínimos ni seleccionar una fila.

`claves_ambiguas_a.csv` conserva todas las filas ambiguas y sus columnas originales.
`procedencia_dataset_a_analitico.csv` conserva archivo y número de registro de
**cada fila con clave válida**, con su clasificación; permite rastrear también
filas únicas y todos los miembros de duplicados exactos. El número de registro
es la posición de datos dentro del CSV, no la línea física cuando hay multilíneas.
El JSON conserva SHA-256 de todos los CSV RAW, D procesado y los `.joblib` presentes,
verificados antes y después. No se escribe en esas fuentes.

## Cantidades y variables candidatas

Se incluyen `CA_CAMAS`, `CA_MEDICOS_TOTAL`, `CA_MEDICOS_RESIDENTES`, `CA_ENFERMERAS`.
Valores vacíos, códigos como `NE_0001`/`NE_0002`, texto inválido y no finitos quedan
`NaN`. No se interpreta su significado ni se convierten en cero. Los negativos
también quedan `NaN` en la copia analítica: el valor negativo original permanece
en hallazgos con motivo `NEGATIVO`. Ceros numéricos genuinos permanecen como cero.
El CSV de calidad contiene hallazgos **por registro RAW y variable**, incluyendo
los que finalmente fueron duplicados o ambiguos; no es un conteo de filas finales.

Únicamente se agregan estas seis variables:

| Variable | Fórmula |
|---|---|
| medicos_por_cama | CA_MEDICOS_TOTAL / CA_CAMAS |
| enfermeras_por_cama | CA_ENFERMERAS / CA_CAMAS |
| residentes_por_cama | CA_MEDICOS_RESIDENTES / CA_CAMAS |
| variacion_camas_a_1m | CA_CAMAS(t) − CA_CAMAS(t−1) |
| variacion_medicos_1m | CA_MEDICOS_TOTAL(t) − CA_MEDICOS_TOTAL(t−1) |
| variacion_enfermeras_1m | CA_ENFERMERAS(t) − CA_ENFERMERAS(t−1) |

Ratios: `NaN` cuando camas <= 0 o un operando no es utilizable. Variaciones:
misma IPRESS y mes anterior calendario exacto; nunca saltan un mes ausente o
ambiguo. No se usa `t+1`, desplazamiento hacia el futuro ni relleno de huecos.

**Disponibilidad:** para predecir riesgo de `t+1`, A(t) solo es admisible si ya
estaba disponible al ejecutar la predicción. El mes de referencia no prueba
fecha de publicación ni descarta revisiones posteriores. Esta condición sigue
pendiente de confirmar; el candidato no equivale a validación de uso operativo.

## Cruce experimental

`D LEFT JOIN A` únicamente por IPRESS, año y mes observado. Se valida
`many_to_one`, se conserva exactamente la cantidad y orden de D y se comprueba
unicidad IPRESS + servicio_hospitalizacion + periodo_actual antes y después.
También se comprueba coherencia entre periodo_actual y año/mes. Duplicados D,
claves A repetidas o columnas A ya presentes hacen fallar el proceso; no se reparan.

`tiene_datos_a=1` significa **coincidencia con una clave A no ambigua**, incluyendo
duplicados exactos resueltos. No significa que los cuatro recursos sean numéricos
ni que los ratios o variaciones estén completos. La cobertura informa por
separado cuántas filas tienen los cuatro recursos numéricos.

Sin coincidencia o con clave ambigua: `tiene_datos_a=0`, todas las covariables A
y estado de calidad A quedan nulos, pero se conserva la fila D. A aporta contexto
institucional mensual repetido en los servicios de esa IPRESS; no personal o
camas exclusivos de cada servicio. No se toca el target ni los indicadores D.

## Artefactos

- `data/processed/dataset_a_analitico.csv`: clave, cuatro recursos, estado de
  calidad y seis derivadas; exactamente una fila por clave no ambigua.
- `data/processed/dataset_modelo_ipress_con_a_experimental.csv`: copia D más A.
- `data/quality/dataset_a/duplicados_exactos_tratados.csv`.
- `data/quality/dataset_a/claves_ambiguas_a.csv`.
- `data/quality/dataset_a/calidad_dataset_a_analitico.csv`.
- `data/quality/dataset_a/cobertura_dataset_a_analitico.csv`: año observado y global.
- `data/quality/dataset_a/comparacion_camas_a_d_detallada.csv`.
- `data/quality/dataset_a/procedencia_dataset_a_analitico.csv` (trazabilidad adicional).
- `data/quality/dataset_a/resumen_preparacion_dataset_a.json`.

Reejecutar actualiza solamente estos artefactos de preparación, no los informes
anteriores de auditoría. `es_modelo_produccion=false` figura en el JSON.

## Resultados locales

| Medida | Resultado |
|---|---:|
| Filas RAW A | 387189 |
| Filas A analíticas | 359856 |
| Claves con duplicado exacto | 18109 |
| Filas repetidas exactas eliminadas | 25835 |
| Claves ambiguas excluidas | 473 |
| Filas RAW pertenecientes a claves ambiguas | 1498 |
| Filas con clave inválida | 0 |
| D antes / después | 45318 / 45318 |
| D con A no ambigua | 44849 |
| Cobertura | 98.9651% |
| D sin A utilizable | 469: 464 sin coincidencia y 5 con clave ambigua |
| D con los cuatro recursos numéricos | 44843 |

Reconciliación: **387189 − 25835 − 1498 = 359856**. Las cinco filas D que
la auditoría anterior consideró ambiguas siguen siendo ambiguas: no eran casos
de repetición exacta. Seis filas con coincidencia válida tienen algún recurso
no numérico; no se convirtieron en ceros.

## Comparación exploratoria de camas por IPRESS-mes

Se calcula suma y máximo de total_camas **solo entre servicios presentes en D
procesado**, sin alterar D ni añadir esas agregaciones como features. Si algún
servicio tiene camas inválidas, no se presenta una suma parcial como total válido.

| Comparación con CA_CAMAS | Suma camas D | Máximo camas D |
|---|---:|---:|
| Pares numéricos válidos | 5810 | 5810 |
| Coincidencia exacta | 43.46% | 12.38% |
| Diferencia dentro de ±10% (5747 pares A>0) | 67.22% | 15.00% |
| Mediana de error absoluto porcentual | 1.70% | 65.77% |
| Correlación Pearson descriptiva | 0.9565 | 0.7139 |

La suma se aproxima más a A en 4585 IPRESS-mes, el máximo en 124 y empatan en
1101. Esto es **más coherente con A como capacidad institucional y D como camas
por servicio**, pero no demuestra que sean intercambiables. D está filtrado por
alcance, calidad y disponibilidad de pareja objetivo; puede omitir servicios y
podrían existir servicios solapados. No se exige igualdad ni se reemplaza ninguna
variable del modelo. La comparación es descriptiva, sin afirmación causal.

## Pruebas y límites de verificación

21 pruebas nuevas aprobadas en `tests/test_preparar_dataset_a.py`, con datos
artificiales: duplicados exactos/ambiguos incluso fuera de las cuatro covariables,
trazabilidad entre archivos, inválidos/negativos, divisiones por cero, continuidad
mensual, independencia de valores futuros, LEFT JOIN sin multiplicación, claves
y periodos incoherentes, comparación separada y preservación de RAW/D/modelos.

Suite ejecutable en este entorno: **287 passed, 1 warning**, excluyendo
`tests/test_entrenar_modelo.py`. Se intentó ejecutar la suite completa, pero la
recolección falla al importar scikit-learn: su extensión local es para Python
3.13 y el ejecutor disponible aquí es Python 3.12. El Python de `.venv` devuelve
«Acceso denegado» desde este entorno. El warning es la deprecación preexistente
Starlette/httpx. No se modificaron dependencias ni se afirma que pasó la suite
completa. La preparación real sí terminó y verificó hashes de todas las fuentes.

Sigue pendiente confirmar fecha de disponibilidad de A(t) y definir el manejo
experimental de covariables faltantes antes de evaluar modelos. Además, la
preparación D existente busca CSV directamente en `data/raw`; el traslado previo
a `data/raw/Hospitalizacion` requerirá un ajuste separado cuando se regenere D.
Este módulo no cambia esa preparación ni su ruta.
