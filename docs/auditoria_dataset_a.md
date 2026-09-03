# Auditoría exploratoria de ConsultaA

Esta fase no incorpora A al dataset de entrenamiento, no entrena modelos y no
modifica archivos de origen. Módulo: `src/auditar_dataset_a.py`.

## Fuentes y alcance del cruce

La búsqueda recursiva se limita al repositorio. Detecta ConsultaA/ConsultaD por
nombre y también A renombrado por sus cabeceras CA_CAMAS y personal. Excluye
entornos, cachés y reportes para no volver a auditar salidas.

Fuentes encontradas: A en `data/raw/Capacidad`, D1 en
`data/raw/Hospitalizacion`. La auditoría conserva las rutas y SHA-256.

El denominador del cruce es **`data/processed/dataset_modelo_ipress.csv`**:
filas D por IPRESS, servicio y mes después del filtro del proyecto, tratamiento
y selección de parejas t->t+1. No son todas las filas nacionales del RAW D.
No se regenera ni modifica D. La columna `anio` de la cobertura es el año del
mes observado t, no el año objetivo usado por el backtesting.

## Lectura, claves y duplicados

Lectura de CSV como texto, con detección de coma/punto y coma y UTF-8/Latin-1.
Se documentan columnas exactas, equivalencias y cambios de esquema por archivo.
Se reconocen explícitamente CO_IPRESS/CODIGO_IPRESS, ANHO/ANIO/AÑO y MES.
No se elige silenciosamente entre dos columnas equivalentes presentes.

El código se normaliza solo en memoria: strip, 1-8 dígitos, relleno a ocho con
ceros, aceptación del sufijo decimal `.0` de exportaciones numéricas. No se
truncan códigos de más de ocho dígitos ni se resuelven códigos por nombre.
Año entero válido (1900 hasta el año de ejecución), mes entero 1-12. Se conservan
valores originales y número de registro de datos (no número físico de línea si
el CSV contiene campos multilínea). Las claves inválidas se reportan.

Se comprueba codigo_ipress+anio+mes sobre todos los A. Las filas repetidas quedan
en `duplicados_clave_a.csv` sin sumarse. Se comparan todas las columnas originales
para identificar diferencias de sede, categoría, servicio u otros campos.
Incluso duplicados exactos se consideran ambiguos hasta revisión.

## Calidad, vista y temporalidad

Por año se cuentan nulos, valores no numéricos, ceros y negativos de camas,
médicos, residentes y enfermeras, junto con descriptivos y percentiles.
Los negativos permanecen en el informe: no se silencian ni corrigen. Cadenas
con separadores numéricos ambiguos quedan como no numéricas; no se adivinan miles.

La vista analítica exige departamento=LIMA, provincia=LIMA y un sector del
conjunto público existente en `datos_raw.SECTORES_PUBLICOS`. La condición
hospitalaria se respalda por presencia del código en D, cuyo alcance es
hospitalización/cuidados críticos 24/25. No se inventa una clasificación de
categorías. Se inspeccionan las categorías reales y se registran sus frecuencias.
Esta vista no identifica hospitales sin actividad en el D procesado.

Los cambios de personal/camas solo se calculan con claves A únicas y meses
consecutivos. Se marca como alerta exploratoria un cambio absoluto >=10 y
relativo >=100%, o cambio desde cero de esa magnitud. No son umbrales oficiales,
no excluyen filas ni corrigen valores. La lista se guarda en `cambios_mensuales_a.csv`.

Candidatas: médicos/cama, enfermeras/cama y residentes/cama usan camas A >0 y
personal válido no negativo; en otro caso NaN. Las variaciones son diferencias
absolutas t menos t-1, no tasas inventadas. Se evalúa calculabilidad, no desempeño.

Un cruce futuro solo podrá usar **A(t) o anteriores para D(t)**. Esta condición
de fecha de referencia no demuestra disponibilidad real: debe confirmarse que
el archivo/valor estaba publicado al cierre de t y no fue revisado usando futuro.
Nunca se utiliza A(t+1) ni se completa un mes con valores posteriores.

## LEFT JOIN y comparación de camas

El cruce conserva todas las filas de D. Primero se une la cantidad de filas A
por clave, y solo para claves únicas se adjuntan variables A. Una coincidencia
ambigua no multiplica D ni proporciona una suma inventada de camas/personal.
La cobertura distingue coincidencia cualquiera, coincidencia única utilizable
y ambigüedad. IPRESS sin coincidencia significa sin ningún mes coincidente del
año; los meses aislados faltantes se reportan por separado.

Se distinguen códigos ausentes, años ausentes y meses sin match. A es nacional
y D está filtrado: muchos A_sin_D son diferencias de alcance, no errores de código.
Un mismo nombre con códigos distintos solo se registra como pista, sin resolverlo.

Camas: comparar cada servicio D con CA_CAMAS A **solo es descriptivo**. No se suman
servicios. La variación de camas entre servicios es evidencia de una granularidad
diferente, no una prueba definitiva de semántica. Se reportan:

- diferencia = D-A; diferencia absoluta = abs(D-A);
- diferencia porcentual = 100*(D-A)/A, únicamente A>0;
- coincidencia exacta, tolerancias 5%/10% y correlación Pearson cuando hay variación;
- tamaños de muestra y ejemplos de discrepancias.

Una A institucional repetida en varios servicios pondera más esas IPRESS en los
descriptivos; no interpretar correlación ni coincidencia como validación de
intercambiabilidad. No se reemplaza total_camas ni días-cama disponibles por A.

## Ejecución y salidas

```powershell
python -m pytest -q
python -m src.auditar_dataset_a
```

Los siete archivos solicitados se generan en `data/quality/dataset_a/`.
Se añaden dos evidencias: `cambios_mensuales_a.csv` y `vista_analitica_a.csv`.
El JSON registra fuentes, granularidad, cobertura, calidad, candidatas,
limitaciones, recomendación y verificación de hashes antes/después.
Si no existe A legible, las métricas quedan no evaluables (null), no cobertura 0%.

**Pendiente del pipeline existente:** el lector de preparación actual busca CSV
directamente en `data/raw`, sin recorrer subcarpetas. Tras ubicar D en
`data/raw/Hospitalizacion`, una próxima preparación requerirá ajustar esa ruta
en una tarea separada. Esta auditoría sí busca recursivamente; no se modificó
la preparación ni se mezcló A con D.

## Resultados verificados en los archivos locales


ConsultaA: 387,189 filas, 12 archivos, años 2015–2026; 26 columnas y sin cambios de esquema.

Clave: 360,329 claves distintas, 18,582 repetidas; máximo 46 filas por clave.

Claves repetidas exactas: 18,109; con diferencias: 473.

Cobertura global D procesado: 98.9761%; unívoca 98.9651%. 464 filas D sin coincidencia y 5 ambiguas.

Las 74 IPRESS D tienen al menos una coincidencia; no implica cobertura de todos sus meses.

Coincidencias D con A fuera de la vista analítica propuesta: 0.


| Año observado | Filas A | Filas D | D con A | Cobertura |
|---|---:|---:|---:|---:|
| 2015 | 619 | 1828 | 1496 | 81.84% |
| 2016 | 4689 | 3755 | 3701 | 98.56% |
| 2017 | 9981 | 4249 | 4249 | 100.00% |
| 2018 | 29951 | 3923 | 3923 | 100.00% |
| 2019 | 33958 | 4088 | 4088 | 100.00% |
| 2020 | 40835 | 3930 | 3853 | 98.04% |
| 2021 | 37758 | 4011 | 4011 | 100.00% |
| 2022 | 37923 | 4427 | 4427 | 100.00% |
| 2023 | 40860 | 4588 | 4588 | 100.00% |
| 2024 | 48879 | 4631 | 4630 | 99.98% |
| 2025 | 60586 | 4593 | 4593 | 100.00% |
| 2026 | 41150 | 1295 | 1295 | 100.00% |


| Variable A nacional | Nulos | % nulo | No numéricos |
|---|---:|---:|---:|
| CA_CAMAS | 12 | 0.00310% | 7045 |
| CA_ENFERMERAS | 12 | 0.00310% | 6670 |
| CA_MEDICOS_RESIDENTES | 12 | 0.00310% | 6776 |
| CA_MEDICOS_TOTAL | 12 | 0.00310% | 6699 |


Camas D por servicio vs A: coincidencia exacta 1.61%; dentro de ±5%: 1.73%; dentro de ±10%: 1.93%; correlación descriptiva 0.2264.

No son intercambiables: por ejemplo, código 00008720 en octubre de 2023 reporta A=1545 y D=4 en cirugía oftalmológica; no se sumaron servicios.

Los valores especiales como NE_0002 se reportan como no numéricos, no como cero ni con una interpretación inventada. Confirmar su catálogo.


Recomendación actual: apto_para_experimento_modelo=false. A muestra potencial como covariables institucionales, pero no está listo para uso automático: acordar tratamiento de coincidencias ambiguas/no numéricas, no equiparar camas A con camas del servicio D y confirmar disponibilidad de A(t). Esta auditoría no demuestra mejora predictiva.

Hay cobertura suficiente para estudiar covariables institucionales, pero antes se debe acordar el tratamiento de las cinco filas ambiguas, valores especiales y fecha de disponibilidad. No se ha medido ninguna mejora del modelo.


Validación: 266 pruebas aprobadas en la suite sin test_entrenar_modelo.py; 17 pruebas de auditoría A aprobadas tras el ajuste final. La suite completa está bloqueada al importar scikit-learn en este entorno. Un warning preexistente de Starlette/httpx. Los hashes de los 24 CSV originales A/D y del D procesado fueron verificados antes y después.

En las 473 claves repetidas con diferencias, cambian cantidades de recursos; no aparecen diferencias de categoría, sector, ubicación o nombre que justifiquen sumarlas como sedes distintas. Las 18,109 claves restantes repiten filas exactamente.
