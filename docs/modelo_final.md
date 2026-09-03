# Configuración productiva congelada

XGBoost base existente + las 69 columnas del conjunto D, en el orden registrado
por `seleccion_regla_extension_020.json`. No añade Dataset A ni flujo/permanencia,
no cambia target, indicadores, hiperparámetros ni datos. El módulo experimental
de features se reutiliza sin modificarlo: se promueve exactamente D a producción.

## Entrenamiento y trazabilidad

`python -m src.entrenar_modelo_final --solo-plan` valida hashes, política vigente,
pares t→t+1 y evidencia de selección, sin escribir ni entrenar.
`python -m src.entrenar_modelo_final` ajusta UNA vez el pipeline existente con
todos los registros elegibles del CSV D. Mantiene `sample_weight` balanceado
`N/(3*N_clase)`, random_state=42, n_estimators=300, max_depth=5,
learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective=multi:softprob,
eval_metric=mlogloss, num_class=3, n_jobs=-1. Compara además TODOS los parámetros
del estimador contra el JSON experimental; falla si difieren.

Dataset verificado al implementar: SHA256
`e0e099bfd10a6a9f3b2e296c5608e85faf8b58f1bf5a86fe426f5e3421399e82`;
45.318 filas, entradas 2015-01 a 2026-03, objetivos 2015-02 a 2026-04.
Son cantidades verificadas del plan: no prueban por sí mismas que se haya
generado el nuevo modelo. Solo un entrenamiento terminado escribe
`es_modelo_final_produccion: true`, fecha, huella del joblib, conteos, listas
exactas de periodos/features, pesos, parámetros, versiones de librerías y código.

Se guardan únicamente `models/modelo_ipress.joblib` y `models/model_metadata.json`.
Se verifican las huellas de todos los archivos existentes en data/ y de los
artefactos models/ restantes. No se borran ni sobrescriben informes experimentales,
clases ni la importancia antigua. Las importancias actuales se calculan del
pipeline ajustado y se exponen desde su metadata; no se reutiliza el CSV antiguo.
La metadata anterior completa se conserva bajo `metricas_historicas_no_vigentes`.

La publicación usa archivos temporales, valida la serialización y revierte el
par anterior ante una excepción de reemplazo. El hash modelo/metadata permite
rechazar una pareja incompleta (también ante interrupción abrupta). No es una
transacción de dos archivos ante un corte de energía. Detener la API durante
el reemplazo y reiniciarla después; los artefactos se mantienen en cache.

## Decisión y probabilidades

Validar `classes_` numéricas 0=bajo, 1=medio, 2=alto, reordenar `predict_proba` y:

1. P(Alto) >= 0.35: Alto.
2. Si no, argmax=Bajo y P(Alto) >= 0.20: Medio.
3. En otro caso: argmax.

No hay fallback a `predict()` ni probabilidades inventadas si falta predict_proba.
`probabilidad` es P(clase final): para [.55,.25,.20] se devuelve Medio y .25.
Se conservan las tres probabilidades originales, sin calibración. El índice
legado `riesgo_insuficiencia_capacidad` mantiene su transformación por compatibilidad
del contrato, pero se documenta en OpenAPI y /metadata como índice visual/operativo
derivado, NO probabilidad calibrada de insuficiencia. No decide ni mide desempeño.

## Evidencia de evaluación, no una nueva evaluación

La regla se eligió con 2018, 2021, 2022, 2023 y 2024. 2025 es comprobación adicional;
**no holdout virgen**. Se leen los valores exactos de los CSV de extensión 020,
no aproximaciones del texto ni métricas del modelo de junio. No se repite 2025.
La metadata distingue resultados de desarrollo y de comprobación. Esas métricas
corresponden a los ajustes por fold; no son una evaluación independiente del
artefacto final entrenado ahora con todos los datos (incluido 2025 y posteriores).

## Inferencia y límites conservados

FastAPI calcula las mismas features con `agregar_features_candidatas`, incluida
la etiqueta ACTUAL observada para conteos retrospectivos; nunca usa el objetivo
futuro como input. Se conservan los nombres públicos y los campos de respuesta.
El wrapper del joblib acepta exactamente las columnas D y su orden. El cliente
API continúa enviando los datos agregados originales y su historial validado.

Las nuevas ventanas exigen continuidad; ante huecos permanecen NaN, que XGBoost
ya manejaba en D. Las variables base conservan promedios parciales/tendencias cero
del experimento. Cinco meses previos cubren ventanas de seis; la racha Alto puede
requerir toda la historia hasta su interrupción. Se permite enviar más de doce
meses y se advierte si toda la historia recibida está en Alto, pues la racha puede
estar truncada. No se inventan meses anteriores.

El CSV D ya omite meses sin pareja objetivo; este límite del historial de
entrenamiento no se corrige aquí. La validación de calidad de nuevos archivos,
las interpretaciones operativas antiguas del ratio y cambios en los consumidores
Java/Angular permanecen fuera de este trabajo. El cliente debe suministrar
historial previamente validado. No hay despliegue, commit, push ni merge.
