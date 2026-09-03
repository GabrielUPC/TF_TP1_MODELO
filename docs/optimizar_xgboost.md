# Optimización temporal limitada de XGBoost

Módulo independiente: `src/optimizar_xgboost.py`. No cambia features, target,
preparación, reglas de calidad, umbrales ni el entrenamiento de producción.
No escribe `.joblib` ni `model_metadata.json`.

## Espacio fijado antes de evaluar

Tres perfiles completos, no un producto cartesiano gigante:

| Parámetro | Base | Regularizado | Profundo |
|---|---:|---:|---:|
| n_estimators | 300 | 200 | 400 |
| max_depth | 5 | 3 | 6 |
| learning_rate | 0.05 | 0.08 | 0.03 |
| min_child_weight | 1 | 3 | 5 |
| subsample | 0.8 | 0.9 | 0.75 |
| colsample_bytree | 0.8 | 0.9 | 0.75 |
| gamma | 0 | 0.1 | 0.2 |
| reg_alpha | 0 | 0.1 | 0.3 |
| reg_lambda | 1 | 3 | 5 |

Cada perfil se cruza con cinco esquemas: balanceado estándar y pesos absolutos
(Bajo, Medio, Alto) = (1,1,1.1), (1,1,1.2), (1,1,1.3), (1,1,1.5).
Total: **15 configuraciones**, incluido XGBoost base con balanceado.
`random_state=42`; el resto de parámetros procede de la fábrica existente.
No se usa `scale_pos_weight`. Se pasa `modelo__sample_weight` al pipeline.
Balanceado = n_train / (3 * frecuencia de la clase en train). Los otros esquemas
son pesos absolutos, no multiplicadores del balanceado. Nunca se cuentan las
clases del test para construir los pesos.

## Separación temporal

El tuning recibe únicamente filas cuyo periodo objetivo es anterior a 2025.
Reutiliza la elegibilidad de backtesting (24 meses históricos distintos, año
objetivo completo, al menos dos casos por clase en train). Con el dataset actual:

- Tuning: **2018, 2021, 2022, 2023 y 2024**; 75 ajustes de evaluación XGBoost.
- Holdout: **2025**, después de congelar la elección.
- 2026 y cualquier año posterior: excluidos por completo.

Cada fold entrena exclusivamente con periodos objetivo anteriores al año evaluado.
El preprocesador se crea y ajusta de nuevo solo con train. No hay validación
aleatoria ni early stopping sobre 2025. La evaluación supone observaciones
mensuales de t disponibles al cierre del mes, como el backtesting existente.

**2025 ya se observó en el backtesting anterior**, por lo que no se presenta como
un holdout nunca visto durante todo el proyecto. En esta búsqueda queda excluido
de toda selección, ranking, ajuste de pesos o elección de hiperparámetros.

## Rankings y elección

Principal: maximizar F1 macro promedio, después balanced accuracy promedio,
después Recall Alto promedio y finalmente minimizar FNR Alto promedio.
Se descarta cualquier candidato con caída de Recall Alto superior a **0.01
absoluto** respecto a la base sobre los mismos folds históricos.

Seguridad: maximizar Recall Alto, minimizar FNR Alto y desempatar por F1 macro
y balanced accuracy, siempre con F1 macro >= base - **0.02 absoluto**.
Se reporta esta alternativa, pero no se usa 2025 para decidir entre rankings.
La configuración del **ranking principal** es la única candidata optimizada
que se evalúa en el holdout. Se permite que la base gane; no se garantiza mejora.
En empate completo se prefiere la base, luego ID de configuración determinista.
Los promedios pesan cada fold igual; la desviación estándar del resumen es
muestral (ddof=1). Se rechazan rankings con métricas principales ausentes.

El JSON contiene la elección, alternativa de seguridad, ambos criterios,
comparaciones históricas contra base y persistencia, y las métricas de 2025
contra XGBoost base, persistencia y Random Forest. Si gana la base, se reutiliza
su evaluación de holdout y no se ajusta dos veces el mismo modelo. La alternativa
de seguridad no se evalúa adicionalmente en 2025. El holdout no modifica la elección.

## Comandos y archivos

Plan sin ajustes:

```powershell
python -m src.optimizar_xgboost --solo-plan
```

Ejecutar tuning y la evaluación final una sola vez, sin producción:

```powershell
python -m src.optimizar_xgboost
```

Se verifica el SHA-256 del dataset contra su metadata. Salidas en `models/`:

- `plan_tuning_xgboost.json`: espacio y años; no contiene resultados simulados.
- `resultados_tuning_xgboost.csv`: métricas por configuración/año, más persistencia.
- `resumen_tuning_xgboost.csv`: medias, dispersión, diferencias contra base, filtros
  de admisión y posiciones en ambos rankings.
- `mejor_configuracion_xgboost.json`: elección congelada y luego resultados holdout,
  con `es_modelo_final_produccion=false`.
- `evaluacion_holdout_xgboost_2025.csv`: comparación final y métricas de Alto.
- `holdout_xgboost_2025.json`: registro persistente de inicio/finalización.

Todas las configuraciones usan las mismas filas de validación; se conserva una
huella del test. Métricas y denominadores de Alto->Bajo se reutilizan del backtesting.
El script no importa los promedios de seis años existentes para escoger parámetros:
recalcula las comparaciones únicamente en los folds históricos de tuning.

Si existe una selección o un registro de holdout previo, el comando se detiene.
No borra esos archivos ni incluye una opción automática para repetir 2025.
Si falla tras congelar la selección o durante el holdout, conserva la evidencia
y exige revisión antes de continuar; no reajusta hiperparámetros basándose en ello.
Las pruebas usan motores artificiales que registran llamadas, sin entrenar producción.
