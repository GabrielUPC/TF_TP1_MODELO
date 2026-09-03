# Reglas de decisión sobre XGBoost D

Experimento separado en `src/evaluar_reglas_decision.py`. No altera features D,
hiperparámetros, target, umbrales del target, preparación, calidad, API ni
producción. Usa la fábrica XGBoost BASE del experimento D y los mismos pesos
balanceados calculados exclusivamente con las clases del train de cada fold.

## Espacio fijo: doce reglas

| Regla | Decisión |
|---|---|
| argmax | Clase con mayor probabilidad |
| alto_0.25 / 0.30 / 0.35 / 0.40 / 0.45 | Si P(Alto) >= umbral, Alto; si no, argmax |
| proteccion_0.20 / 0.25 / 0.30 / 0.35 | Si argmax=Bajo y P(Alto) >= umbral, Medio; si no, argmax |
| combinada_0.35_0.25 | Si P(Alto)>=0.35, Alto; si no, si argmax=Bajo y P(Alto)>=0.25, Medio; si no, argmax |
| combinada_0.40_0.25 | Si P(Alto)>=0.40, Alto; si no, si argmax=Bajo y P(Alto)>=0.25, Medio; si no, argmax |

Solo se prueban esas dos combinaciones; no se buscan más umbrales. Los límites son
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

Antes de aplicar las reglas se diagnostican las probabilidades originales de
cada año de desarrollo. No se ajusta un calibrador ni se utilizan etiquetas de
2025 para calibración o selección:

- Log-loss: media de `-log(p_clase_real)`. Solo para evitar log(0) se limita el
  argumento del log a [1e-15,1]; la matriz original permanece intacta.
- Brier multiclase: media por caso de la **suma** de `(p_c - indicador_real_c)^2`
  para las tres clases, rango [0,2]. No se divide por tres. Se informa también
  Brier binario por clase.
- Confianza media máxima: media de `max(p_bajo,p_medio,p_alto)`, junto a accuracy
  argmax y su diferencia. Esa diferencia sola no prueba buena calibración.
- Reliability OVR por clase: diez intervalos fijos [0,.1), ..., [.9,1], con
  número de casos, probabilidad media y frecuencia real observada. Intervalos
  vacíos conservan n=0 y medias NaN. ECE por clase = suma ponderada por n_bin/N
  del error absoluto entre esas medias; ECE macro = media de las tres clases.

Alerta exploratoria fijada antes de evaluar: con al menos 100 casos en el año,
ECE de alguna clase >=0.10 o diferencia absoluta confianza−accuracy >=0.10.
Es una señal descriptiva para revisar, **no un umbral clínico, prueba estadística
ni certificado de calibración**. Sin alerta tampoco se asegura calibración,
especialmente con pocos casos. La alerta se documenta y no ajusta probabilidades
ni cambia automáticamente el ranking. Ante desajuste, la regla continúa siendo
un experimento, no una política validada para producción.

Se reutilizan todas las métricas del backtesting: accuracy, balanced accuracy,
F1 macro, métricas Alto, falsos negativos y errores Alto->Bajo, entre otras.
`proporcion_alto_bajo` = errores Alto->Bajo / casos Alto reales del fold.
También se conserva la proporción respecto al total de casos bajo otro nombre.
ROC-AUC permanece igual entre reglas porque la matriz de probabilidades no cambia.
Se reportan conteos y proporciones predichas de Bajo/Medio/Alto por regla y año.
El resumen conserva conteos acumulados y medias de proporciones con igual peso
por año. Marca como riesgoso un aumento de proporción predicha Alto >=0.10
absoluto frente a argmax (10 puntos porcentuales), por año y en promedio.
Esta convención exploratoria es una alerta, no exclusión automática ni límite
de capacidad hospitalaria. Precision Alto y F1 Alto permiten observar el costo.

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

Versión actual `reglas_decision_D_calibracion_12_v3`: prioridad de errores severos,
12 reglas y diagnóstico de calibración. Sustituye el ranking FNR-primero de la
versión v2 conforme al pedido nuevo. Se comparan tasas con igual peso por año,
conservando los conteos absolutos; no se altera el entrenamiento ni se incorpora A.

Se distingue `admisible` (cumple F1) de `elegible_seleccion`:

- No elegir una regla que no mejore ninguna de las dos tasas de error frente
  a argmax; argmax siempre permanece como referencia seleccionable.
- Veto por beneficio mínimo: si **ambas** reducciones promedio (Alto->Bajo y FNR)
  son menores a 0.005 absoluto, y la caída de F1 o balanced accuracy es mayor a
  0.01, la regla no es seleccionable aunque cumpla el límite de F1 de 0.02.
- Esos valores significan 0.5 puntos porcentuales y 1 punto porcentual. Son
  convenciones operacionales exploratorias declaradas antes del experimento,
  no resultados optimizados usando 2025 ni criterios clínicos.

El CSV y JSON guardan motivos de exclusión, alertas de expansión, comparación
explícita contra argmax y ranking solo de reglas seleccionables.

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

- `models/resultados_reglas_decision.csv`: 60 filas de desarrollo (12 x 5) más
  1 o 2 filas de comprobación 2025, diferenciadas por `fase`.
- `models/resumen_reglas_decision.csv`: solo desarrollo, con medias, variabilidad,
  diferencias respecto a argmax, admisibilidad y ranking.
- `models/seleccion_regla_decision.json`: reglas, años, criterios, elección,
  promedios, comparación argmax, comprobación 2025, procedencia y
  `es_modelo_final_produccion=false`.
- `models/calibracion_probabilidades.csv`: 150 filas (5 años x 3 clases x
  10 bins). Métricas globales del año se repiten en sus bins para facilitar
  revisión; no sumarlas ni contarlas como observaciones independientes.
  Incluye huellas de test y probabilidades. No contiene 2025.

El JSON se crea de forma exclusiva antes de consultar 2025. Si cualquiera de
estas salidas ya existe, el comando se detiene sin borrar ni sobrescribir. Esto
impide repetir automáticamente 2025, también ante ejecuciones concurrentes. Un
fallo posterior a la selección conserva resultados parciales y estado; requiere
revisión, no borrar evidencias o reintentar a ciegas.

Las pruebas usan probabilidades y motores artificiales; sus resultados no
constituyen evidencia de mejora real de XGBoost. Se necesita ejecutar la evaluación
con el entorno Python compatible del proyecto para obtener una regla ganadora.

Las pruebas de v3 verifican también combinaciones inclusivas y prioridad de Alto,
Brier/log-loss con valores conocidos, bins vacíos y probabilidad 1, alerta sin
recalibrar, veto de mejora mínima, expansión de Alto y diagnóstico anterior a
las reglas. La verificación de ejecución real y tests se informa al finalizar;
ninguna métrica artificial de los tests se guarda como resultado real en models.

Verificación local v3: **32 tests del evaluador aprobados**. Suite ejecutable:
**327 passed, 1 warning**, excluyendo `tests/test_entrenar_modelo.py` por
incompatibilidad del entorno; el warning preexistente corresponde a Starlette/httpx.
El plan real validó las doce reglas y los cinco años. La ejecución real falló
antes del primer ajuste: `.venv` devuelve «Acceso denegado» y el ejecutor alternativo
Python 3.12 no carga la extensión sklearn instalada para Python 3.13. Los 77
archivos de datos/artefactos preexistentes conservaron sus hashes. No se generaron
todavía los cuatro reportes reales, no se evaluó 2025 y no hay conclusión real
sobre calibración ni regla seleccionada en esta versión.

## Extensión mínima con protección 0.20 (sin nuevo ajuste)

`reglas_extension_020()` define exactamente seis reglas, sin cambiar las doce
del experimento anterior: argmax, proteccion_0.20, alto_0.40, alto_0.35,
combinada_0.40_0.20 y combinada_0.35_0.20. En las combinadas primero se aplica
P(Alto)>=0.40/0.35; solo si no se cumple se protege argmax=Bajo con P(Alto)>=0.20.

`evaluar_extension_020(df, probabilidades_por_anio, referencia)` recibe D original,
un diccionario {2018: matriz, 2021: matriz, 2022: matriz, 2023: matriz, 2024: matriz}
y `resultados_reglas_decision.csv` leído como DataFrame. Las matrices deben tener
columnas Bajo/Medio/Alto y exactamente el orden de registros de cada test previo.
Se exigen coincidencias de `probabilidades_sha256` y `test_sha256` con la referencia.
Si cambia matriz, orden, test o etiquetas, aborta. No llama fit, predict,
predict_proba ni importa el motor de entrenamiento; no escribe archivos.

Devuelve 30 filas de desarrollo, resumen de seis reglas y selección histórica.
Conserva el criterio de selección y límite F1 previo. `reduce_ambos_errores` indica
reducción simultánea de FNR y proporción Alto->Bajo respecto a argmax; se informa
el delta exacto de F1, sin inventar otro umbral de «prácticamente estable».

Solo después de guardar/revisar esa selección se usa
`comprobar_extension_020_2025(df, probabilidades_2025, referencia, regla_seleccionada)`.
Verifica también las huellas originales de 2025 y compara únicamente argmax con
la seleccionada. No ajusta modelos. Las dos funciones separan explícitamente la
selección histórica de la comprobación, que no debe usarse para reescoger la regla.

**Bloqueo encontrado:** los resultados anteriores ya existen, pero el evaluador
que los produjo conservó métricas y hashes, no las matrices por registro. Un hash,
un resumen de calibración o una matriz de confusión no permiten recuperar esas
probabilidades. Se requiere el archivo o la matriz en memoria de la ejecución
original. No se ha reentrenado para recrearlas ni se han sobrescrito resultados.
El comando habitual `python -m src.evaluar_reglas_decision` sigue correspondiendo
al experimento anterior: **no usarlo para esta extensión**.

Validación de la extensión: 35 tests del evaluador aprobados (tres nuevos),
incluyendo espacio exacto de seis reglas, límites inclusivos, prioridad de Alto,
preservación de probabilidades, rechazo de hashes diferentes y prohibición de
obtener probabilidades nuevas. Falta la evaluación real de las dos combinaciones
hasta disponer de las matrices originales.
