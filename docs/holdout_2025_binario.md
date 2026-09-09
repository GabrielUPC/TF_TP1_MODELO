# Holdout sellado 2025 del modelo binario

Las decisiones congeladas antes de abrir el holdout son XGBoost, threshold
principal 0.36 y sensibilidades 0.32/0.28. Random Forest usa 0.37 y Regresión
Logística 0.45 como comparaciones preespecificadas desde desarrollo 2018-2024.

La ejecución tiene dos pasos. `preauditar` carga únicamente features, periodos e
identificadores; comprueba que train termine en target 2024-12, que test contenga
solo targets 2025 y que 2026 quede fuera. Después `evaluar` ajusta los tres
modelos con train, genera scores de test sin etiquetas y recién entonces abre
una vez las etiquetas 2025. Una segunda evaluación en el mismo directorio se
rechaza.

La evaluación informa discriminación, calibración cruda, decisiones congeladas,
familias 24xx/25xx y drift. Los intervalos de confianza usan bootstrap por
cluster IPRESS+servicio, 1,000 repeticiones y semilla 20250905. Ningún estimador
se guarda y los artefactos productivos permanecen intactos.

## Ejecución reproducible

La salida debe estar fuera del repositorio y comenzar vacía. Primero se ejecuta
únicamente la auditoría sin etiquetas:

```powershell
python -m src.evaluar_holdout_2025_binario `
  --modo preauditar `
  --dataset <dataset-candidato.csv> `
  --metadata <dataset-metadata.json> `
  --escenarios-desarrollo <escenarios-operacionales.csv> `
  --probabilidades-desarrollo <probabilidades-desarrollo.csv> `
  --output-dir <directorio-HOLDOUT_2025_BINARIO>
```

Solo si `auditoria_temporal_holdout_2025.json` indica
`AUDITORIA_PREAPERTURA_APROBADA`, se permite la apertura única:

```powershell
python -m src.evaluar_holdout_2025_binario `
  --modo evaluar `
  --dataset <dataset-candidato.csv> `
  --metadata <dataset-metadata.json> `
  --escenarios-desarrollo <escenarios-operacionales.csv> `
  --probabilidades-desarrollo <probabilidades-desarrollo.csv> `
  --output-dir <mismo-directorio-HOLDOUT_2025_BINARIO>
```

El segundo comando se niega a sobrescribir una evaluación existente. La
verificación final registra hashes de los CSV, mantiene 2026 sin predecir ni
evaluar y declara que 2025 no intervino en la selección del modelo o threshold.
