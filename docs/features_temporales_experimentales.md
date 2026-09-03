# Experimento temporal Base/A/B/C/D (v2)

Código aislado en `variables_temporales_experimentales.py` y
`evaluar_features_temporales.py`. No cambia `COLUMNAS_PREDICTORAS`, preparación,
indicadores, target, reglas Q05-Q09, API ni artefactos de producción.

## Inspección y variables reutilizadas

La base contiene métricas observadas de t (ingresos, egresos, pacientes-día,
camas, días-cama, etc.), indicadores calculados con t, calendario/categorías,
cinco promedios móviles de tres meses y cuatro tendencias de un mes. Las
variables históricas originales usan pasado y verifican continuidad. El
objetivo usa el mes siguiente exclusivamente para construir la etiqueta;
ninguna etiqueta futura se incorpora al conjunto predictor.

No se duplican los siguientes cálculos:

| Nombre solicitado | Columna BASE reutilizada |
|---|---|
| cambio_ocupacion_1m | tendencia_ocupacion_1m |
| cambio_ingresos_1m | tendencia_ingresos_1m |
| cambio_egresos_1m | tendencia_egresos_1m |
| promedio_ocupacion_3m | promedio_movil_3m_ocupacion |

Se mantienen sus contratos: tendencias cero ante huecos y promedio parcial
cuando no hay tres meses. No se añaden copias con otro tratamiento de NaN.
`max_presion_ingresos_3m` reutiliza el nombre experimental `max_presion_3m`;
no se generan ambas columnas. Las columnas duplicadas del experimento previo
se retiran solo de los conjuntos experimentales, sin tocar el dataset.

## Variables nuevas y conjuntos

32 columnas nuevas, agrupadas de forma fija, sin combinaciones arbitrarias:

| Conjunto | Añadidas a BASE | Composición acumulada |
|---|---:|---|
| Base | 0 | Features actuales intactas |
| A | 11 | Lags |
| B | 15 | A + cambios no duplicados y aceleración |
| C | 22 | B + historia reciente no duplicada |
| D | 32 | C + riesgo reciente, márgenes y crecimientos |

- Lags: ocupación 1/2/3m; pacientes-día, días-cama disponibles, ingresos y
  egresos 1/2m. Cada lag es el valor exacto de t-k y exige todos los meses
  intermedios por `codigo_ipress + servicio_hospitalizacion`.
- Cambios nuevos: ocupación 2m, pacientes-día 1m y días-cama 1m = actual menos
  lag correspondiente. Aceleración = ocupación(t) - 2*ocupación(t-1) + ocupación(t-2).
- Historia: máximo, mínimo y desviación de ocupación 3m (ddof=0); promedio de
  ocupación 6m; máximo de presión de ingresos 3m; promedios de pacientes-día y
  días-cama disponibles 3m.
- Riesgo: conteos de Alto 3/6m, Medio+Alto 3m y racha consecutiva de Alto hasta t.
  La racha se corta en huecos, datos ausentes o clase distinta de Alto; ante Alto
  tras un hueco comienza en 1. Si la clase actual falta, la racha es NaN.
- Márgenes: ocupación - 0.85; ocupación - 0.70; abs(ocupación - 0.85).
- Crecimientos: (valor(t) - valor(t-1)) / valor(t-1), para pacientes-día y
  días-cama. Solo se divide cuando previo > 0 y actual >= 0, ambos finitos.
  Denominador cero/negativo, faltantes y resultados no finitos producen NaN,
  sin inventar crecimiento cero. Son fracciones, sin multiplicar por 100.
  Brecha = crecimiento_demanda_1m - crecimiento_capacidad_1m.

Los márgenes son transformaciones deterministas/colineales de ocupación, no
información independiente; se conservan como transformaciones solicitadas en D,
y se documentan como posibles redundancias. No se garantiza beneficio con XGBoost.
El JSON incluye fórmula, fuentes, grupo y conjuntos de cada variable nueva.

## Continuidad y ausencia de información futura

Lags y ventanas nuevas solo usan t y pasado. Las ventanas incluyen t y requieren
3 o 6 meses consecutivos completos, sin atravesar huecos. Quedan NaN cuando la
historia no alcanza. No hay desplazamientos negativos ni ventanas centradas.
Los conteos de riesgo consultan únicamente la clase ACTUAL observada. Las clases
actual/futura en bruto quedan fuera del pipeline predictor.

NaN se conserva: StandardScaler se ajusta exclusivamente al train de cada fold
y conserva ausentes; XGBoost los admite. No hay imputación con test o futuro.
Se crea un pipeline nuevo por ajuste, con los parámetros XGBoost BASE de la
fábrica existente y pesos balanceados calculados solo en train. No se importan
los ganadores del tuning anterior. Cada variante conserva las mismas filas de
prueba; no se eliminan casos según disponibilidad de sus nuevas features.

Limitaciones metodológicas:

- El dataset procesado omite meses sin pareja t+1: existe menos historia que
  antes de construir el objetivo. No se recuperan meses desde RAW ni pendientes.
- Se supone disponibilidad de las métricas de t al cierre. No se han validado
  fechas reales de publicación ni revisiones retrospectivas de las fuentes.
- Historial y márgenes están correlacionados con otras variables; importancia
  baja no demuestra ausencia de información, ni importancia alta causalidad.
- Los años de desarrollo ya fueron usados en experimentos previos; sigue
  existiendo riesgo de sobreajuste por selección reiterada.

## Evaluación y selección fijadas

Desarrollo: **2018, 2021, 2022, 2023, 2024**, según año del periodo OBJETIVO.
Train contiene solo periodos objetivo anteriores al año evaluado. Los cinco
folds deben cumplir la elegibilidad del backtesting existente. Se filtra 2025
y posteriores antes de generar features y consultar etiquetas para desarrollo.

25 ajustes: cinco variantes por cinco años. Se calculan todas las métricas del
backtesting, incluidas las de Alto, Alto->Bajo y ROC-AUC macro. Se conserva la
huella de las filas de test. Los años pesan igual en las medias; desviación
poblacional, mínimo y máximo siguen el resumen del backtesting.

Regla de selección:

1. Rechazar candidatos con caída de F1 macro > 0.02 absoluto frente a Base.
2. Prioridad 1: candidatos que mejoren F1 macro o balanced accuracy.
3. Solo si no hay candidatos admisibles de prioridad 1, prioridad 2: mejorar
   Recall Alto o reducir FNR Alto (sin imponer el 0.01 del experimento previo).
4. Dentro de cada prioridad: F1 macro, balanced accuracy, Recall Alto y menor
   FNR, en ese orden. Empates exactos: A, B, C, D. Sin mejoras admisibles: Base.

Se usa tolerancia numérica 1e-12. Se marca explícitamente cuando Recall mejora
pero F1 macro o balanced accuracy empeoran. Nunca se favorece D por su tamaño.
La elección queda escrita antes de la ablación y de 2025.

## Importancia y ablación diagnóstica

Se reajusta solo el ganador en cada uno de los cinco folds históricos para
calcular permutation importance: caída de F1 macro al permutar cada columna
original del test, tres repeticiones, semilla 42, máximo 1,000 filas elegidas
al azar del test del fold. No se ajusta el modelo al permutar. El CSV registra
año, feature, media/desviación de la caída, tamaño de test/muestra y método.
No se calcula importancia en 2025 ni se interpreta como causalidad.

Después se retiran, individualmente, `ratio_camas_disponibles`,
`presion_ingresos_camas` y `anio`, reajustando sobre los mismos folds. Las
columnas geográficas con importancia F1 media histórica <= 0 se retiran juntas
en una cuarta ablación como máximo (departamento/provincia/distrito). Es un
diagnóstico exploratorio, no una eliminación definitiva ni prueba estadística.

Se reportan métricas y diferencias frente al ganador original por año. Los
derivados permanecen: retirar presión no retira automáticamente su historial.
La ablación no cambia el ganador, las features de producción ni la elección
que se comprobará en 2025. Máximo 20 ajustes de ablación y cinco de importancia.

## Comprobación adicional 2025 y archivos

Tras congelar la selección y realizar los diagnósticos históricos se evalúan
una vez Base y el conjunto elegido en 2025, entrenando solo con años objetivo
anteriores a 2025. Si gana Base, se hace un único ajuste y se reutilizan las
métricas de referencia. Se ignora 2026 por completo.

**2025 ya se observó antes: NO es un holdout virgen.** No interviene en selección,
importancia ni ablación, y sus resultados no alteran ninguna elección.

Salidas (no contienen modelos serializados):

- `models/resultados_features_temporales.csv`: fases `desarrollo`, `ablacion`,
  `comprobacion_2025`; métricas por variante/año y diferencias de ablación.
- `models/resumen_features_temporales.csv`: solo Base/A/B/C/D históricos;
  métricas agregadas, prioridad, admisibilidad, tradeoff y selección.
- `models/seleccion_features_temporales.json`: definición/linaje, criterios,
  selección congelada, comparación contra Base, ablaciones, comprobación 2025,
  riesgos y `es_modelo_final_produccion=false`.
- `models/importancia_features_temporales.csv`: importancia del ganador solo
  en pruebas históricas. No incluye 2025.

Si existe cualquiera de esas salidas, se detiene sin sobrescribir. La reserva
exclusiva del JSON evita que ejecuciones concurrentes repitan 2025. Si un paso
falla después de congelar la elección, los resultados parciales y estado se
conservan; se requiere revisión, no borrar archivos ni reintentar a ciegas.

```powershell
python -m src.evaluar_features_temporales --solo-plan
python -m src.evaluar_features_temporales
python -m pytest -q
```

El plan comprueba el dataset contra su metadata SHA-256 y no escribe resultados
ni ajusta modelos. La ejecución completa tiene hasta 52 ajustes de evaluación;
ninguno entrena/guarda el modelo final de producción. Las pruebas usan motores
artificiales; no sustituyen la evaluación real de desempeño con XGBoost.
