# Modelo binario: fase experimental

La variable científica es `alta_presion_capacidad_siguiente_mes`: vale 1 únicamente
si `ocupacion_siguiente_mes > 0.90` y vale 0 si es menor o igual a 0.90. Una
ocupación de 0.90 exacto pertenece a la clase 0. La versión metodológica es
`alta_presion_capacidad_t1_v1`.

El evento se interpreta como **condición futura de alta presión sobre la
capacidad de camas**. La ocupación observada es
`total_pacientes_camas / total_camas_disponibles`, donde el denominador son
días-cama disponibles. D1 no observa directamente demanda rechazada, por lo que
esta etiqueta no demuestra falta real de camas ni insuficiencia asistencial.

## Construcción y compatibilidad

Se construye `X_t -> ocupacion_siguiente_mes -> target binario`. El siguiente mes
debe existir exactamente en el calendario para la misma IPRESS y servicio. Un
mes pendiente o ausente rompe la relación: no se une t con t+2. La normalización
IPRESS precede a la agrupación y ordenación; el código conserva ocho caracteres.

Las columnas multiclase se conservan temporalmente como **legacy** para mantener
disponible la implementación productiva anterior. Ninguna etiqueta legacy ni la
ocupación futura se utilizan como predictores o criterios de selección binarios.

| Archivo o módulo | Dependencia legacy | Tratamiento en esta fase |
|---|---|---|
| `src/preparar_dataset.py` | Etiqueta observada y desplazamiento multiclase | Agrega ocupación futura y target binario; documenta las etiquetas antiguas como compatibilidad |
| `src/entrenar_modelo.py` | Tres clases, pesos balanceados y métricas macro/OVR | Se conserva; el experimento binario usa su propio protocolo |
| `src/backtesting_temporal.py` | Folds de tres clases, recall Alto y errores Alto→Bajo | Se conserva como evidencia histórica; no evalúa el target nuevo |
| `src/variables_temporales.py` | Variables mensuales disponibles hasta t | Se reutilizan con continuidad exacta |
| `src/variables_temporales_experimentales.py` | Features D, algunas derivadas de las categorías 70/85% | La lista binaria excluye las derivadas de esas categorías |
| `src/entrenar_modelo_final.py`, `src/modelo_final.py` | Entrenamiento y carga del modelo multiclase productivo | No se ejecutan ni sustituyen |
| `src/main.py` | Contrato FastAPI productivo Bajo/Medio/Alto | Se conserva intacto |
| `src/evaluar_reglas_decision.py` | Reglas de decisión multiclase y métricas Alto | Se conserva; no se reutilizan sus thresholds en el experimento binario |

## Predictores y familias

`src/features_binarias.py` define una lista explícita de 62 predictores:
las 69 features D anteriores menos cuatro conteos derivados de las categorías
legacy y tres márgenes frente a 0.70/0.85. Se mantienen variables de t y de su
historia disponible: ocupación, ingresos, egresos, estancias, pacientes-día,
días-cama disponibles, camas, indicadores, medias móviles, tendencias,
estacionalidad y descriptores del establecimiento/servicio.

Las variables redundantes pueden seguir como candidatas a revisión; no se
eliminan selectivamente después de ver el resultado. El artefacto
`auditoria_features.csv` documenta cada columna, origen, momento disponible,
riesgo de información futura y decisión de inclusión.

Quedan excluidos `ocupacion_siguiente_mes`, el target binario, todas las etiquetas
legacy, `periodo_predicho` e identificadores usados exclusivamente para
trazabilidad. `familia_servicio`, derivada de `id_hospitalizacion`, sirve solo para
evaluación: prefijo 24 = `hospitalizacion`; prefijo 25 = `cuidados_criticos`.
`servicio_hospitalizacion` permanece como descriptor disponible en t. No se
entrenan modelos separados por familia en esta fase.

## Protocolo temporal

El entrenamiento de cada fold utiliza exclusivamente periodos objetivo
anteriores al año evaluado. Se requieren al menos 24 meses históricos distintos
y ambas clases en entrenamiento; el año de evaluación debe tener 12 meses.
Los huecos globales no invalidan por sí solos un fold: la continuidad se exige
en cada observación y sus variables temporales.

Solo los años completos y elegibles anteriores a 2025 participan en desarrollo.
**2025 queda reservado y no se calculan sus métricas predictivas en esta fase.**
2026 es parcial y se excluye del desarrollo. `plan_folds_binario.csv` registra
los años elegibles y los motivos de exclusión; los años se calculan a partir del
dataset candidato, sin imponer resultados.

Logistic Regression, Random Forest y XGBoost se comparan con los mismos
registros, features y preprocesamiento, ajustado dentro de cada train. Los
baselines son la clase mayoritaria del train y la persistencia de presión:
`ocupacion_estimada_t > 0.90`. No se emplean pesos de clase, `sample_weight`,
SMOTE, remuestreo ni selección extensa de hiperparámetros. Los pesos balanceados
del código multiclase no se heredan silenciosamente.

Cada modelo produce `P(Y_(t+1)=1)`. La clasificación utiliza 0.50 como **umbral
técnico provisional para evaluación**, no como umbral operacional definitivo.
No se traducen las probabilidades a Bajo/Medio/Alto.

Se calculan ROC-AUC binario, Average Precision (PR-AUC), recall, specificity,
precision, F1, balanced accuracy, accuracy, Brier Score, TP/TN/FP/FN y
prevalencias real y predicha, por año y familia. Las métricas sin clases
suficientes se reportan como no aplicables. La comparación considera PR-AUC,
ROC-AUC, variabilidad anual, recall, Brier, familias y falsos negativos; no se
declara un ganador de producción por accuracy.

La calibración se mide en crudo con Brier y una tabla de calibración. No se
aplica calibración sigmoid/isotonic ni se seleccionan thresholds operacionales.
Los targets >85% y Q90 histórico quedan como sensibilidad metodológica, sin
sustituir el target principal.

## Ejecución y evidencia externa

La preparación permite escribir el candidato fuera del repositorio. Usar una
carpeta externa con permiso de escritura:

```powershell
python -m src.preparar_dataset --output-dir D:\Carrera\Cursos\TP1\BACKUP\AUDITORIA_MODELO_BINARIO\dataset
python -m src.backtesting_binario --dataset D:\Carrera\Cursos\TP1\BACKUP\AUDITORIA_MODELO_BINARIO\dataset\dataset_modelo_ipress.csv --metadata D:\Carrera\Cursos\TP1\BACKUP\AUDITORIA_MODELO_BINARIO\dataset\dataset_metadata.json --output-dir D:\Carrera\Cursos\TP1\BACKUP\AUDITORIA_MODELO_BINARIO
```

Antes de ajustar candidatos se recalculan los controles: 48,363 meses antes del
target, 46,042 pares exactos, 15,847 positivos y 30,195 negativos; prevalencia
esperada 34.418574%, cero saltos y cero duplicados IPRESS-servicio-mes. Estos son
controles auditados, no resultados que deban forzarse. Una discrepancia detiene
la evaluación para explicar la causa.

La evidencia experimental incluye `auditoria_features.csv`,
`plan_folds_binario.csv`, `metricas_backtesting_binario.csv`,
`resumen_modelos_binario.csv`, `metricas_24xx_25xx.csv`,
`matrices_confusion_binario.csv` y `resumen_metodologico.md`. Se guarda fuera del
repositorio y no se incorpora a Git en esta fase. Este documento describe el
protocolo; no contiene métricas inventadas ni declara que un candidato ya fue
evaluado.

Los RAW, `models/modelo_ipress.joblib`, `models/model_metadata.json`, la evidencia
productiva histórica y FastAPI permanecen protegidos. No se ejecuta
`entrenar_modelo_final.py` ni se reemplaza el modelo productivo.
