# Tratamiento posterior a la auditoría

Versión: **`capacidad_q05_q06_q07_q08_v3`**.

Si una fila RAW en alcance tiene Q05, Q06, Q07 o Q08, queda pendiente todo el grupo
`codigo_ipress + servicio_hospitalizacion + anio + mes`. La clave se normaliza
igual que en la preparación; no depende del nombre del hospital ni del archivo.
Se aparta antes de consolidar: una fila válida no puede esconder el problema al
sumarse con otra del mismo grupo. No se excluyen hospitales o años completos.

No se corrigen/intercambian columnas, imputan valores ni modifican CSV originales.
Q07 es una hipótesis fuerte que exige validación. Q08 mantiene la definición
pacientes-día / días-cama disponibles > 1,20, con denominador positivo, y su
severidad REVISAR en auditoría. El tratamiento ahora aparta su grupo completo
si la fila está en alcance; fuera de alcance se audita sin apartar grupos del
modelo. El valor exacto 1,20 no activa Q08. Q09 solo se reporta y no provoca
exclusión por sí solo. Q00 sigue bloqueando archivos; Q04 se limita a
auditar duplicados, aunque la limpieza heredada mantenga su deduplicación.

## Flujo integrado

```text
RAW → auditoría Q00–Q09 → lectura de archivos con esquema válido
    → limpieza heredada y filtro de alcance
    → apartar grupos Q05/Q06/Q07/Q08
    → consolidación → indicadores → variables temporales → target
    → dataset procesado y metadata
```

Las definiciones de consolidación, indicadores, percentiles, variables temporales
y target permanecen iguales. Al apartar febrero, enero no puede usarlo como
etiqueta, febrero no puede ser entrada y marzo no puede incorporar febrero a su
historial. No se conecta enero con marzo saltando el hueco.

## Evidencia

En `data/quality/` se generan:

- `pendientes_capacidad.csv`: hallazgos Q05/Q06/Q07/Q08 en alcance, valores fuente y
  estado `PENDIENTE_VALIDACION_NO_USAR_ENTRENAMIENTO`.
- `meses_pendientes_capacidad.csv`: claves únicas de los grupos apartados.
- `tratamiento_capacidad.json`: versión, fecha UTC, reglas, conteos y huellas.

`reglas_aplicadas` contiene `["Q05", "Q06", "Q07", "Q08"]`;
`reglas_solo_auditadas` contiene `["Q01", "Q02", "Q03", "Q04", "Q09"]`.
Se distinguen filas RAW con hallazgos, grupos pendientes y filas retiradas antes
de consolidar (estas últimas se cuentan después de la limpieza existente).

`data/processed/dataset_metadata.json` conserva sus campos anteriores y añade el
tratamiento, el SHA-256 del dataset y las huellas de fuentes. La preparación
comprueba las fuentes antes y después; no deben editarse durante su ejecución.
Si no quedan parejas de meses consecutivos, falla sin reemplazar el dataset
anterior. Los reportes de la ejecución fallida permanecen para revisión.

## Uso y límites

La auditoría aislada no prepara ni entrena. Para regenerar el dataset usando la
nueva política, cuando se decida hacerlo:

```powershell
python -m src.preparar_dataset
```

Este cambio no regenera automáticamente el dataset real, no ejecuta entrenamiento
y no modifica `entrenar_modelo.py`. Por tanto, **el entrenamiento existente aún
puede leer un dataset antiguo**: deberá prepararse de nuevo antes de entrenar.
La metadata no representa por sí sola un bloqueo de entrenamiento ni una
certificación de calidad integral.

FastAPI, backend y frontend conservan su comportamiento actual. Esta política
Q05/Q06/Q07/Q08 se aplica en la preparación; no añade un bloqueo Q08 a la API.
La auditoría aislada sigue reportando Q08 como REVISAR y no aparta datos por sí
sola. La cantidad de ejemplos y los valores derivados pueden cambiar al
reconstruir la preparación, sin cambiar sus fórmulas.
