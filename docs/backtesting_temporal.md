# Backtesting temporal expansivo

Este módulo evalúa estabilidad temporal sin seleccionar ni guardar un modelo de
producción. No llama a `entrenar_modelos`, `guardar_resultados`, `joblib.dump` ni
al test aleatorio. Reutiliza los algoritmos, pipelines y pesos de clase de
`entrenar_modelo.py`; cada fold crea y ajusta su propio preprocesador solo con train.

## Elegibilidad y tiempo

La partición usa **periodo_predicho**, no el año de las features:
train = todos los periodos objetivo anteriores al año; test = únicamente ese año.
Un año completo requiere presencia de enero a diciembre en el conjunto de test,
no doce registros por cada IPRESS. No se eliminan registros de test según la clase.
Por defecto se requieren los 24 meses objetivo inmediatamente anteriores al año,
sin huecos globales, y al menos dos ejemplos de cada clase en entrenamiento.
Son criterios mínimos configurables, no una garantía de suficiencia estadística.
El plan registra años aceptados y descartados, motivos, tamaños y clases de train.
Se rechazan periodos que no correspondan exactamente a t+1, índices duplicados y
periodos duplicados de una IPRESS/servicio. Los índices originales se conservan.

Las variables temporales usan t y hasta t-2, con continuidad exacta. Usar meses
anteriores del mismo año de test como historia de una predicción posterior supone
una evaluación mensual con observaciones ya disponibles; no es pronosticar todo
el año de una vez desde enero. Se supone disponibilidad de los datos al cierre
del mes t. Retrasos reales de reporte requerirían otra evaluación de disponibilidad.
Las cuatro columnas objetivo y las excluidas no pueden entrar como predictores.

## Ejecución separada de producción

Solo documentar años (sin ajustar modelos):

```powershell
python -m src.backtesting_temporal --solo-plan
```

Evaluar modelos por fold, sin entrenar el modelo final de producción:

```powershell
python -m src.backtesting_temporal
```

Se comprueba que el SHA-256 del dataset coincida con su metadata. Las opciones
`--min-meses-historial` (24) y `--min-casos-clase` (2) quedan registradas en la
comparación. Una dependencia ausente o un fallo de un modelo detiene la evaluación;
no se ocultan folds fallidos ni se comparan subconjuntos de años diferentes.
XGBoost solo se incluye si la fábrica existente lo ofrece. No se modifica su configuración.

## Salidas en models/

- `plan_backtesting_temporal.csv`: elegibilidad y motivos por año, sin ajustar modelos.
- `metricas_backtesting_temporal.csv`: una fila por algoritmo/baseline y año.
- `resumen_backtesting_temporal.csv`: promedio, desviación estándar poblacional
  (ddof=0), mínimo, máximo y número de valores válidos por métrica y modelo.
- `comparacion_backtesting_temporal.json`: criterios, mejores resultados (incluidos
  empates), disponibilidad de XGBoost y, desde CLI, huella del dataset y definición
  de target. No reemplaza `model_metadata.json` ni certifica al modelo desplegado.

Cada fila de métricas registra tamaños y una huella común del test. Todos los
algoritmos y los tres baselines (mayoría de train, persistencia y regla de ocupación)
usan exactamente las mismas filas y etiquetas. La mayoría se calcula solo en train.
Con el target vigente persistencia y regla de ocupación son equivalentes; ambas
se conservan para mantener visible esa equivalencia.

## Métricas

Accuracy, balanced accuracy, precision macro, recall macro, F1 macro,
especificidad macro y ROC-AUC OVR macro. Las macros usan las clases fijas 0/1/2;
balanced accuracy promedia recall de clases presentes, como la métrica habitual.
Las divisiones indefinidas por ausencia de predicciones usan cero en precision.
ROC-AUC utiliza probabilidades en orden 0/1/2 y el criterio de rangos OVR (empates
promediados). Queda vacío si faltan clases reales o probabilidades; no se inventan
probabilidades para los baselines. Se valida forma, rango y suma de probabilidades.

Además: recall, precision y F1 de Alto, casos Alto reales, falsos negativos Alto,
FNR Alto, cantidad Alto->Bajo y dos proporciones:

- `tasa_falsos_negativos_alto` = Alto predicho Bajo o Medio / Alto real.
- `proporcion_alto_bajo` = Alto predicho Bajo / Alto real.
- `proporcion_alto_bajo_total` = Alto predicho Bajo / total de test.

Sin Alto real, recall/F1/FNR Alto y la proporción condicionada a Alto quedan vacíos,
no como rendimiento perfecto. El resumen ignora valores ausentes e informa cuántos
valores válidos aportaron al promedio. Cada año pesa igual; no se mezclan todos
los registros para ocultar variaciones entre años.

## Comparación

Orden lexicográfico transparente: maximizar F1 macro promedio; en empate,
maximizar balanced accuracy promedio; luego Recall Alto promedio; finalmente
minimizar FNR Alto promedio. Valores ausentes quedan al final. Se reportan todos
los empates exactos, sin usar el nombre como criterio. Se informa el mejor modelo
y también el mejor resultado incluyendo baselines. Esto es comparación de
backtesting, no una selección automática de artefacto para producción ni una
estimación independiente posterior a la selección. No se ajustan hiperparámetros
con los años de prueba en este módulo.

El comando histórico de entrenamiento queda intacto y conserva su comportamiento
anterior; no ejecutarlo si se desea únicamente backtesting. El análisis aleatorio
existente sigue siendo secundario y no participa en esta comparación.

## Verificación local y resultados pendientes

Con el dataset inspeccionado (45.318 filas), los años elegibles por defecto son
2018, 2023, 2024 y 2025. El plan guardado detalla los descartes: 2015/2026 y
2019/2020 no tienen los doce meses objetivo; 2016/2017 aún no reúnen historial
suficiente; los huecos objetivo previos excluyen 2021/2022 bajo el mínimo
conservador de 24 meses consecutivos. No se rellenaron periodos ni se cambió
la preparación para ampliar el número de folds.

Se probaron particiones, invariancia frente al futuro, métricas, orden de
probabilidades, comparación y exportación con datos artificiales y modelos de
prueba que registran las llamadas. No se ajustaron modelos reales ni se guardó
un modelo de producción: el intento real falló al importar scikit-learn
(instalado para Python 3.13, intérprete disponible Python 3.12).
Por ello solo se generó el plan real; los CSV de métricas/resumen se verificaron
en directorios temporales de pruebas y sus resultados reales quedan pendientes.
No hay un ganador real determinado en esta ejecución.
