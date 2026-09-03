# Reglas de decisión sobre XGBoost D

Experimento separado en `src/evaluar_reglas_decision.py`. No altera features D,
hiperparámetros, target, umbrales del target, preparación, calidad, API ni
producción. Usa la fábrica XGBoost BASE del experimento D y los mismos pesos
balanceados calculados exclusivamente con las clases del train de cada fold.

## Espacio fijo: diez reglas

| Regla | Decisión |
|---|---|
| argmax | Clase con mayor probabilidad |
| alto_0.25 / 0.30 / 0.35 / 0.40 / 0.45 | Si P(Alto) >= umbral, Alto; si no, argmax |
| proteccion_0.20 / 0.25 / 0.30 / 0.35 | Si argmax=Bajo y P(Alto) >= umbral, Medio; si no, argmax |

No se combinan ambas familias ni se buscan más umbrales. Los límites son
inclusivos. La matriz de probabilidades se valida y ordena como Bajo/Medio/Alto
usando `classes_`. Empates argmax: primera clase en ese orden. Nunca se alteran
las probabilidades: no se normalizan, recalibran ni sustituyen por puntuaciones.

Por cada fold se ajusta **un** pipeline XGBoost D y se llama una sola vez a
`predict_proba`. Todas las reglas usan esa matriz y las mismas filas. Las salidas
registran la huella del test y de la matriz. Las probabilidades no están guardadas
en los informes agregados del experimento anterior; por eso se necesita un ajuste
de evaluación por año para obtenerlas, no un modelo diferente por regla.

## Separación temporal

Selección exclusiva con años objetivo **2018, 2021, 2022, 2023, 2024**. Train es
siempre anterior al año evaluado. Se reutilizan el generador de features D y la
continuidad temporal del experimento anterior sin modificar sus fórmulas.

2025 y posteriores se excluyen antes de construir features de desarrollo o
consultar sus etiquetas. Solo después de escribir la regla seleccionada se
obtienen las probabilidades de 2025. En esa comprobación se ajusta una sola vez
XGBoost D con años objetivo anteriores a 2025 y se comparan argmax y la regla
congelada sobre las mismas probabilidades. Si gana argmax, basta una fila de
resultados y el JSON reutiliza esa referencia para ambas comparaciones.

Total: cinco ajustes históricos + uno para 2025, nunca un ajuste por regla.
2026 se excluye por completo. No se carga ni sobrescribe el modelo de producción.

## Métricas y selección

Se reutilizan todas las métricas del backtesting: accuracy, balanced accuracy,
F1 macro, métricas Alto, falsos negativos y errores Alto->Bajo, entre otras.
`proporcion_alto_bajo` = errores Alto->Bajo / casos Alto reales del fold.
También se conserva la proporción respecto al total de casos bajo otro nombre.
ROC-AUC permanece igual entre reglas porque la matriz de probabilidades no cambia.

Los promedios dan el mismo peso a cada año; desviación poblacional, mínimo y
máximo siguen el resumen del backtesting. Se incluye argmax como candidato y
se rechazan comparaciones con distintas filas, probabilidades o años.

Regla admisible: F1 macro promedio >= F1 macro de D con argmax - 0.02 absoluto.
La comparación del límite admite tolerancia numérica de 1e-12. Después se ordena:

1. Menor proporción Alto->Bajo.
2. Menor FNR Alto.
3. Mayor Recall Alto.
4. Mayor F1 macro.
5. Mayor balanced accuracy.

En empate total se prefiere argmax y después el orden fijo de reglas. No se
impone una mejora ficticia ni un umbral adicional de precisión no solicitado;
precision_alto y f1_alto se reportan para revisar el costo de cada decisión.

## Límites metodológicos

- La protección Bajo->Medio reduce errores severos pero no convierte casos en
  predicciones Alto: por sí sola no mejora Recall Alto ni FNR Alto.
- La promoción a Alto puede mejorar sensibilidad y empeorar precisión. Cumplir
  el límite de F1 no equivale a demostrar seguridad clínica.
- Los umbrales son decisiones experimentales; no implican probabilidades
  calibradas ni cambian la definición de riesgo hospitalario del target.
- Los años históricos se reutilizan tras otros experimentos: existe riesgo
  de sobreajuste de selección. Se mantienen las limitaciones del historial D.
- **2025 ya fue observado**. Es una comprobación adicional, no un holdout virgen.
  Sus resultados nunca alteran la regla seleccionada.
- No se publica esta política en `/predict` ni se entrena producción.

## Salidas y ejecución

```powershell
python -m src.evaluar_reglas_decision --solo-plan
python -m src.evaluar_reglas_decision
python -m pytest -q
```

`--solo-plan` verifica el dataset/metadata SHA-256 y los años sin ajustar modelos
ni escribir resultados. La ejecución completa genera:

- `models/resultados_reglas_decision.csv`: 50 filas de desarrollo (10 x 5) más
  1 o 2 filas de comprobación 2025, diferenciadas por `fase`.
- `models/resumen_reglas_decision.csv`: solo desarrollo, con medias, variabilidad,
  diferencias respecto a argmax, admisibilidad y ranking.
- `models/seleccion_regla_decision.json`: reglas, años, criterios, elección,
  promedios, comparación argmax, comprobación 2025, procedencia y
  `es_modelo_final_produccion=false`.

El JSON se crea de forma exclusiva antes de consultar 2025. Si cualquiera de
estas salidas ya existe, el comando se detiene sin borrar ni sobrescribir. Esto
impide repetir automáticamente 2025, también ante ejecuciones concurrentes. Un
fallo posterior a la selección conserva resultados parciales y estado; requiere
revisión, no borrar evidencias o reintentar a ciegas.

Las pruebas usan probabilidades y motores artificiales; sus resultados no
constituyen evidencia de mejora real de XGBoost. Se necesita ejecutar la evaluación
con el entorno Python compatible del proyecto para obtener una regla ganadora.
