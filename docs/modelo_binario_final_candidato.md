# Modelo binario final candidato v1

Este artefacto predice `P(Y_(t+1)=1)`, donde `Y_(t+1)=1` significa que la
ocupación observada del mismo servicio e IPRESS será mayor que 90% en el mes
calendario siguiente exacto. Una probabilidad de 0.70 significa una estimación
de 70% para ese evento; no significa una ocupación de 70%.

La alerta operacional cumple una regla fija:

```text
alerta_alta_presion = probabilidad_alta_presion >= 0.36
```

La salida contiene `probabilidad_alta_presion`, `threshold` y
`alerta_alta_presion`. El modelo es binario y no produce clases Bajo, Medio o
Alto. Esos niveles, si se crean posteriormente, serán una capa de presentación
separada y no clases aprendidas por este estimador.

## Entrenamiento y evidencia

XGBoost usa las 62 features binarias aprobadas, en el orden guardado en la
metadata. Sus hiperparámetros permanecen congelados: 300 estimadores,
profundidad 5, learning rate 0.05, subsample 0.8, colsample 0.8, objetivo
`binary:logistic`, métrica `logloss`, semilla 42 y cuatro trabajos. No se aplica
un calibrador nuevo, búsqueda adicional, pesos de clase ni remuestreo.

La selección provino del backtesting temporal 2018–2024. Posteriormente, 2025
se abrió una sola vez como holdout sellado: ROC-AUC 0.9333, PR-AUC 0.8983,
Brier 0.0944 y, con threshold 0.36, recall 86.01%, especificidad 84.22%,
precisión 74.36%, F1 0.7976, 240 FN y 509 FP. Una vez terminada esa validación,
las etiquetas válidas hasta 2025-12 se incorporan a este nuevo entrenamiento
candidato. Por ello, esas métricas describen la versión previa al reajuste y no
son una evaluación independiente del joblib final reajustado.

2026 permanece cerrado: no se entrena, predice, evalúa ni utiliza para tomar
decisiones. El candidato se guarda separado como
`models/modelo_ipress_binario_v1.joblib`; no reemplaza el modelo multiclase
legacy ni modifica FastAPI.

## Inferencia local

`src.inferencia_binaria.cargar_modelo_binario()` verifica el SHA-256, metadata,
versión, threshold y lista ordenada de features. Después,
`predecir_alta_presion()` rechaza features faltantes e infinitos. Las columnas
adicionales se rechazan por defecto y solo pueden ignorarse mediante una opción
explícita; en cualquier caso, el DataFrame se reordena al contrato de 62
features antes de predecir.

Durante el entrenamiento, el programa compara tamaño y fecha de modificación de
todos los archivos existentes en `data/` y `models/` antes y después del ajuste.
La comprobación final de integridad vuelve a contrastar los SHA-256 históricos
de RAW y procesados con el manifiesto externo de la auditoría previa.
El joblib se crea, recarga y verifica primero en un directorio temporal fuera
del repositorio. Solo después se copia al nombre candidato y se vuelve a validar
su SHA-256; cualquier fallo elimina los dos archivos candidatos incompletos.

## Limitaciones

La etiqueta representa presión futura observada, no demanda rechazada ni una
prueba causal de insuficiencia de camas. La evaluación 2025 mostró desempeño
algo menor en servicios 25xx: deben monitorearse por separado durante la futura
integración. La frase anterior describía el estado histórico al cerrar el candidato. En la versión actual, el artefacto se expone mediante FastAPI en `/prediccion-binaria`, Spring Boot lo consume desde el flujo productivo `/predicciones-riesgo/binaria/automatica` y Angular invoca ese flujo desde el dashboard; los niveles BAJO/MEDIO/ALTO permanecen como capa de presentación del Backend.
