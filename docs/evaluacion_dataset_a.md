# Experimento temporal D frente a D+A

Módulo independiente: `src/evaluar_dataset_a.py`. No cambia preparación, target,
umbrales, reglas de calidad, hiperparámetros, API ni modelo de producción.

## Variantes fijas

D significa los predictores existentes de `entrenar_modelo.COLUMNAS_PREDICTORAS`
más el **conjunto D de features temporales** del experimento anterior. No significa
la variante Base de aquel experimento. Todas las variantes usan exactamente ese D.

| Variante | Columnas A añadidas |
|---|---|
| D | Ninguna |
| D+A_CAMAS | CA_CAMAS |
| D+A_RECURSOS | CA_CAMAS, CA_MEDICOS_TOTAL, CA_MEDICOS_RESIDENTES, CA_ENFERMERAS |
| D+A_RATIOS | Las cuatro anteriores + medicos_por_cama, enfermeras_por_cama, residentes_por_cama |
| D+A_COMPLETO | Las siete anteriores + variacion_camas_a_1m, variacion_medicos_1m, variacion_enfermeras_1m |

No se prueban combinaciones extra. `tiene_datos_a` **no es predictor**: se usa solo
para verificar y reportar cobertura. A es contexto institucional mensual de la
IPRESS, no camas o personal asignados a cada servicio.

## Fuentes y comprobaciones

Se leen D original, A analítico y D+A experimental, sin regenerarlos. Se verifica
el SHA-256 de D contra su metadata y de las fuentes contra la preparación A.
Se comprueba que A tiene claves únicas, estados UNICA/DUPLICADO_EXACTO y ninguna
clave de `claves_ambiguas_a.csv`. Se recalculan las derivadas A en memoria para
verificar ratios y diferencias t menos t-1 con continuidad mensual exacta.

Se reconstruye en memoria el LEFT JOIN en IPRESS+año+mes **observado t** y se
compara con el CSV experimental: mismos registros, orden, columnas D y covariables
A; unicidad de IPRESS+servicio+periodo_actual. Una inconsistencia aborta, no corrige.
No se usan A(t+1), meses futuros ni imputación. No se eliminan las filas sin A.
NaN se mantiene para el tratamiento nativo de XGBoost; StandardScaler y OneHotEncoder
siguen siendo los del pipeline existente y se ajustan únicamente en train.

La ejecución verifica nuevamente los hashes de fuentes, datasets, metadata y
`.joblib` al terminar, también cuando falla el experimento.

## Evaluación y selección

Años de **periodo_predicho** para desarrollo: 2018, 2021, 2022, 2023, 2024.
Train usa todos los periodos objetivo anteriores al año evaluado; test solo ese
año. Se reutilizan folds, métricas, pesos balanceados calculados con y_train y
pipeline del experimento de features. XGBoost se obtiene de `obtener_modelos()`:
300 estimadores, profundidad 5, learning_rate 0.05, subsample/colsample_bytree 0.8,
random_state 42 y demás parámetros existentes, sin tuning ni early stopping nuevo.
La decisión de clase sigue la predicción estándar del clasificador; no se aplican
las reglas experimentales de promoción/protección de otra tarea.

Las cinco variantes comparten índices de train y test. Cada resultado conserva
`n_test` y `test_sha256`; la selección rechaza años, registros o filas duplicadas
incompatibles. El resumen da media, desviación estándar poblacional, mínimo y
máximo por métrica; cada año pesa igual.

Orden lexicográfico de selección:

1. Mayor F1 macro promedio.
2. Mayor balanced accuracy promedio.
3. Mayor Recall Alto promedio.
4. Menor FNR Alto promedio.
5. Menor proporción Alto→Bajo promedio.

Se excluye una variante si su F1 cae más de 0.02 absoluto frente a D. En empate
total se conserva el orden D, CAMAS, RECURSOS, RATIOS, COMPLETO. No se favorece
más complejidad. Bajo este ranking una variante con menor F1 que D normalmente
no gana, aunque sea admisible: el límite no invierte la prioridad del F1.

La selección se guarda antes de consultar etiquetas de 2025. Si gana D+A, se
retiran individualmente CA_CAMAS, CA_MEDICOS_TOTAL, CA_ENFERMERAS, medicos_por_cama
y enfermeras_por_cama, **solo las presentes en la ganadora**. La ablación se evalúa
en los cinco años históricos y no cambia la selección. Delta positivo de F1 al
retirar una columna sugiere perjuicio en ese experimento; negativo sugiere aporte.
Los derivados correlacionados permanecen: esto no identifica causalidad.

Después se evalúan D y la ganadora una sola vez en 2025. Si gana D, solo se evalúa
D y no se hace ablación. **2025 ya fue observado: no es un holdout completamente
virgen.** No se elige otra variante después de ver este resultado.

Se calculan accuracy, balanced accuracy, precision/recall/F1 macro, especificidad
macro, ROC-AUC OVR macro cuando es evaluable, precision/recall/F1 Alto, casos Alto,
falsos negativos y FNR Alto, errores Alto→Bajo y su proporción. La proporción
principal Alto→Bajo usa como denominador los **casos Alto reales**; también se
conserva la proporción sobre todo el test, claramente separada.

## Ejecución y archivos

Desde la raíz del repositorio, con el entorno Python del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.evaluar_dataset_a --solo-plan
.\.venv\Scripts\python.exe -m src.evaluar_dataset_a
```

`--solo-plan` no ajusta modelos ni escribe reportes. La evaluación real genera:

- `models/resultados_dataset_a.csv`: desarrollo, ablación y evaluación 2025,
  identificados por `fase`; no mezclar fases para seleccionar.
- `models/resumen_dataset_a.csv`: las cinco variantes, solo desarrollo,
  admisibilidad, ranking y diferencias contra D.
- `models/seleccion_dataset_a.json`: selección congelada, variables, parámetros,
  cobertura, procedencia, ablación, evaluación 2025 y limitaciones;
  `es_modelo_final_produccion=false`.

No guarda estimadores. Rechaza ejecución si alguno de esos tres archivos ya
existe, para no sobrescribir evidencia ni repetir automáticamente 2025. Si falla
después de guardar selección, requiere revisar el estado antes de continuar;
no eliminar resultados automáticamente.

## Verificación local y limitaciones

El plan sobre los datos reales se validó: 45318 registros, 44849 con A no ambigua,
469 sin A, cobertura 98.9651%. Los cinco años solicitados son elegibles. Se prevén
25 ajustes de desarrollo, hasta 25 de ablación y hasta dos de evaluación 2025.

La ejecución real se intentó, pero no inició ajustes: `.venv` devuelve «Acceso
denegado» en este entorno; el ejecutor alternativo Python 3.12 no puede importar
la extensión scikit-learn instalada para Python 3.13. No se cambiaron dependencias.
**No hay todavía ganador, métricas comparativas, ablación real ni evaluación 2025
de este experimento. No se generaron CSV/JSON con resultados simulados.**

Los tests usan pequeños datos artificiales y un clasificador de prueba para
verificar el contrato del experimento, no para estimar desempeño real.
Resultado: **302 passed, 1 warning**, con 15 pruebas nuevas y excluyendo
`tests/test_entrenar_modelo.py`, cuya importación bloquea la suite completa por
la incompatibilidad indicada. El warning preexistente corresponde a Starlette/httpx.
Se intentó la suite completa; no se afirma que haya pasado.

Riesgos metodológicos que permanecen:

- A(t) solo es admisible si ya estaba disponible al ejecutar la predicción de
  t+1. Fecha de publicación y revisiones retrospectivas no confirmadas.
- Desarrollo y 2025 fueron usados en experimentos anteriores: riesgo de
  sobreajuste de selección y evaluación no completamente independiente.
- Se conserva el historial D procesado y sus features seleccionadas, incluidas
  transformaciones de riesgo observado; no se añade etiqueta futura al predictor.
- La cobertura por año usa año objetivo, distinto del año observado usado en la
  auditoría A. Coincidencia A no garantiza recursos/ratios completos.
