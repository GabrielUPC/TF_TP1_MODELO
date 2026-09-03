# Experimento de flujo y permanencia frente a XGBoost D

Módulos aislados: `src/variables_flujo.py` y `src/evaluar_features_flujo.py`.
No modifican el dataset, target, umbrales, features D, hiperparámetros, calidad,
API ni modelo de producción. No incorporan Dataset A ni reglas de decisión.

## Definiciones exactas

I = total_ingresos; E = total_egresos; P = total_pacientes_camas (pacientes-día);
S = total_estancias; O = ocupacion_estimada. Todo por IPRESS + servicio.

| Bloque | Variable | Fórmula |
|---|---|---|
| A | balance_flujo_mes | I(t) − E(t) |
| A | balance_flujo_acumulado_3m | suma del balance en t, t−1, t−2 |
| A | promedio_balance_flujo_3m | media del balance en t, t−1, t−2 |
| A | meses_ingresos_mayor_egresos_3m | número de meses I>E en t, t−1, t−2 |
| A | ratio_egresos_ingresos | E(t)/I(t) |
| B | crecimiento_ingresos_1m | (I(t)−I(t−1))/I(t−1) |
| B | crecimiento_ingresos_2m | (I(t)−I(t−2))/I(t−2) |
| B | crecimiento_pacientes_cama_1m | (P(t)−P(t−1))/P(t−1) |
| B | crecimiento_pacientes_cama_2m | (P(t)−P(t−2))/P(t−2) |
| C | estancia_promedio_actual | S(t)/E(t) |
| C | estancia_promedio_lag_1m | estancia_promedio_actual(t−1) |
| C | cambio_estancia_promedio_1m | estancia_promedio_actual(t) − estancia_promedio_actual(t−1) |
| C | promedio_estancia_3m | media de estancia_promedio_actual en t, t−1, t−2 |
| D | aceleracion_ingresos | I(t) − 2I(t−1) + I(t−2) |
| D | aceleracion_pacientes_cama | P(t) − 2P(t−1) + P(t−2) |
| E | racha_ocupacion_creciente_3m | 0 si O(t)<=O(t−1); si sube, 1 + indicador O(t−1)>O(t−2) |
| E | meses_ocupacion_sobre_80_ultimos_3m | número de meses O>0.80 en t, t−1, t−2 |

La racha exige los tres meses válidos. Ejemplos: .70→.76→.82 da 2;
.80→.70→.82 da 1; .70→.82→.82 da 0. No cuenta subidas anteriores si
la última transición no aumenta. El límite >0.80 es estricto y solo describe una
feature: no cambia los umbrales del target.

Los meses se ordenan por grupo; se resta el ordinal calendario para exigir
continuidad. Para t−2 debe existir también t−1. Toda ventana requiere sus tres
valores válidos, sin promedios parciales. División con denominador <=0, datos
no numéricos, no finitos o negativos de origen: NaN en la copia de cálculo.
Los balances y crecimientos negativos válidos se conservan. No se altera el
contenido original, ni se rellena con cero o datos posteriores. Índices y orden
de entrada se conservan. Las claves duplicadas, colisiones y periodos incoherentes
se rechazan en vez de corregirse.

## Cinco variantes, sin búsqueda adicional

| Variante | Nuevas columnas sobre D |
|---|---:|
| D | 0 |
| D+FLUJO (A) | 5 |
| D+DEMANDA (A+B) | 9 |
| D+PERMANENCIA (A+B+C) | 13 |
| D+DINAMICA (A+B+C+D+E) | 17 |

D = predictores base actuales + conjunto temporal D del experimento anterior.
Se preservan sus fórmulas y limitaciones. Las columnas nuevas se calculan solo
en memoria; no se agregan al CSV ni al listado de predictores de producción.

## Evaluación, selección y métricas

Desarrollo por año **objetivo**: 2018, 2021, 2022, 2023, 2024. Train contiene solo
periodos objetivo anteriores al año evaluado. Los cinco modelos comparten los
mismos índices de train/test; las huellas de test permiten verificarlo.
Se reutilizan el pipeline, métricas y pesos balanceados del experimento D.
El escalador y codificador se ajustan exclusivamente en train; NaN se mantiene
para XGBoost. No hay imputación futura ni tuning.

Se usa el XGBoost base de `entrenar_modelo.obtener_modelos()` sin cambiar parámetros
(300 estimadores, max_depth=5, learning_rate=.05, subsample=.8,
colsample_bytree=.8, random_state=42 y demás valores existentes). Las decisiones
son las predicciones estándar, sin umbrales de P(Alto).

Orden lexicográfico: F1 macro promedio, balanced accuracy promedio, Recall Alto
promedio (mayores mejores), FNR Alto y proporción Alto→Bajo (menores mejores).
Cada año tiene igual peso. Admisible si F1>=F1(D)−0.02, tolerancia numérica 1e-12.
En empate completo se usa el orden fijo de la tabla, prefiriendo D. La restricción
no hace ganar una variante con menor F1 que D cuando D sigue en la comparación.

Se reportan accuracy, balanced accuracy, precision/recall/F1 macro, ROC-AUC OVR
macro cuando corresponde, precision/recall/F1 Alto, falsos negativos Alto, FNR
Alto, errores Alto→Bajo y proporción sobre los casos Alto reales. También se
conservan especificidad macro, soporte Alto y proporción severa sobre todo el
test, con nombres separados. Resumen: media, desviación poblacional, mínimo,
máximo y número de años evaluables por métrica.

La elección se escribe antes de acceder a etiquetas 2025. Después se evalúan D
y la seleccionada una vez cada una; si gana D, solo un ajuste. **2025 ya fue
observado, no es holdout virgen.** No se cambia la elección según ese resultado.
No hay ablación ni combinaciones extra en este experimento.

## Archivos y ejecución

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.evaluar_features_flujo --solo-plan
.\.venv\Scripts\python.exe -m src.evaluar_features_flujo
```

`--solo-plan` no ajusta ni guarda artefactos. La ejecución completa crea:

- `models/resultados_features_flujo.csv`: 25 filas de desarrollo y 1 o 2 de
  comprobación 2025, separadas por fase.
- `models/resumen_features_flujo.csv`: cinco variantes, solo desarrollo.
- `models/seleccion_features_flujo.json`: fórmulas, columnas, años, medias,
  comparación con D, criterio, parámetros, procedencia, 2025 y
  `es_modelo_final_produccion=false`.

No guarda modelos. Si cualquiera de esas salidas existe, se detiene sin
sobrescribir ni repetir 2025. Un fallo posterior a la selección deja estado y
resultados parciales para revisión, no se borran automáticamente. D debe coincidir
con el hash de su metadata; se verifican antes/después los CSV y artefactos
preexistentes de models, incluso si falla la evaluación.

## Evidencia local y pendientes

El plan real validó los cinco años sobre el dataset de 45318 filas. Se requieren
25 ajustes históricos y hasta dos de comprobación. Las ventanas completas de
tres meses quedan ausentes en 10.06% del historial de desarrollo; el promedio
de estancia de tres meses en 11.93%. Se mantienen esas filas, con NaN.

La evaluación real se intentó: `.venv` devuelve «Acceso denegado» en este entorno.
El ejecutor alternativo Python 3.12 no puede importar la extensión scikit-learn
local para Python 3.13. No se cambiaron dependencias ni se inventaron métricas.
No hay aún ganador, diferencia frente a D ni evaluación 2025 de este experimento.

Pruebas: 18 nuevas aprobadas. Suite ejecutable: **321 passed, 1 warning**,
excluyendo `tests/test_entrenar_modelo.py` por la incompatibilidad de scikit-learn
indicada. El warning preexistente es de Starlette/httpx. No se afirma que la
suite completa haya pasado. Los tests de evaluación usan un motor artificial
para verificar aislamiento, métricas y selección, no para medir desempeño real.

Recomendación provisional: **no incorporar a producción ni descartar por supuesto
bajo aporte** sin completar la comparación. Hay redundancias conocidas:
balance_flujo_mes repite diferencia_ingresos_egresos; el crecimiento de pacientes-día
de 1m repite crecimiento_demanda_1m en datos válidos; la estancia actual comparte
fórmula con promedio_estancia, con distinta protección ante cero; media y suma
del balance de tres meses son proporcionales. Se conservan porque fueron solicitadas.

Los balances no son un censo ni una medición directa de camas liberadas. La
selección reiterada sobre los mismos años puede sobreajustarse. El historial D
procesado está limitado a meses con pareja objetivo, y la disponibilidad real de
datos al cierre de t sigue siendo una condición de uso. Estas limitaciones no
se corrigen alterando otras áreas del proyecto en este trabajo.
