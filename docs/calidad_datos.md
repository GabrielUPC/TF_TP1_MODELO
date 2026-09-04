# Auditoría de los CSV históricos

La auditoría detecta y registra; no repara, elimina ni sobrescribe archivos RAW.
El tratamiento posterior toma una decisión distinta: apartar los grupos Q05/Q06/Q07/Q08
antes de consolidar. Véase [tratamiento_capacidad.md](tratamiento_capacidad.md).

## Estructura inspeccionada antes del cambio

`src/` contenía `__init__.py`, `entrenar_modelo.py`, `indicadores.py`,
`interpretacion.py`, `main.py`, `predecir.py`, `preparar_dataset.py`,
`soporte_decision.py`, `validar_plantilla.py`, `variables_temporales.py` y
`ver_metricas.py`. No existían `calidad_datos.py` ni `tratamiento_capacidad.py`.

- `preparar_dataset.py` contenía lectura CSV, aliases, esquema, limpieza, filtro
  Lima pública, consolidación y creación del dataset/metadata en `data/processed/`.
- `validar_plantilla.py` valida otra representación de datos (plantilla CSV/Excel),
  campos requeridos, números, periodos y duplicados. No sustituye la auditoría RAW.
- `main.py` valida solicitudes de predicción. No se modifica en esta tarea.
- `entrenar_modelo.py` consume `data/processed/dataset_modelo_ipress.csv` y guarda
  artefactos en `models/`. No se modifica ni se ejecuta en esta tarea.
- Existían siete archivos de pruebas: `test_api.py`, `test_entrenar_modelo.py`,
  `test_indicadores.py`, `test_interpretacion.py`, `test_preparar_dataset.py`,
  `test_soporte_decision.py` y `test_validar_plantilla.py`.

## Ejecución

Desde la raíz del proyecto, usando su entorno Python compatible:

```powershell
python -m src.calidad_datos
```

Genera en `data/quality/`:

- `resumen_calidad.csv`: una fila por archivo, formato, alcance y contadores.
- `hallazgos_calidad.csv`: archivo, fila aproximada, regla, severidad,
  descripción, año/mes, IPRESS, nombre, servicio, alcance y valores originales.
- `resumen_calidad.json`: fecha UTC, criterios, reglas y totales.

Los códigos de salida son 0 para esquemas utilizables (aunque haya alertas),
1 si algún archivo queda bloqueado por Q00 y 2 para errores de ejecución/salida.
El reporte se reemplaza al repetir la auditoría. No ejecutar dos auditorías
simultáneas sobre la misma salida. Se admite `--raw-dir` y `--output-dir` para
trabajar con datos artificiales; la salida no puede estar dentro del RAW.

## Lectura compartida

`datos_raw.py` reúne las funciones y constantes antes incluidas en
`preparar_dataset.py`. Este último las reexporta para mantener sus contratos.
Se conservan aliases, columnas y sectores públicos; el alcance de servicios se
centraliza por los prefijos 24/25 de ID_HOSPITALIZACION.
La auditoría lee con `conservar_originales=True` para no convertir literales como
`NA`, `null` o vacíos en valores distintos. La preparación conserva la lectura
heredada. No se duplican los lectores ni los filtros de alcance.

Se inspeccionan todos los CSV no temporales de
`data/raw/Hospitalizacion`, de todas las regiones
y sectores. `en_alcance_modelo` distingue Lima/Lima, sectores públicos admitidos,
e ID_HOSPITALIZACION cuyo texto, tras quitar espacios extremos, comienza con
24 (hospitalización) o 25 (cuidados críticos). Se conserva 245600, hospitalización
de día. El nombre, sus tildes o su ausencia no incluyen ni excluyen un ID; las
familias 22, 23, 04, 13, 15, 16 y cualquier otra quedan fuera. Se mantienen ceros
iniciales: 0241800 no se transforma en 241800. Auditoría y preparación usan la
misma función. Se normaliza en memoria; los archivos fuente permanecen intactos.

El contexto `codigo_ipress` de los hallazgos utiliza la representación canónica
RENIPRESS de 8 caracteres. Cuando ocurre una transformación,
`valores_relevantes` conserva tanto
`CO_IPRESS_ORIGINAL` como `CO_IPRESS_CANONICO`; el RAW permanece inmutable. La
misma función compartida se usa en preparación y tratamiento para evitar que
una alerta con `9251` deje de coincidir con una fila preparada como `00009251`.

## Reglas

| Regla | Criterio | Consecuencia de la auditoría |
| --- | --- | --- |
| Q00 | Archivo ilegible/vacío, esquema incompleto o encabezados normalizados/aliases ambiguos | ERROR: archivo bloqueado |
| Q01 | Métrica requerida, año o mes vacío, inválido o no finito | ERROR, evidencia del literal |
| Q02 | Métrica negativa | ERROR, sin recorte automático |
| Q03 | Año no entero o fuera de 1900–2100, o mes no entero/fuera de 1–12 | ERROR |
| Q04 | Duplicado exacto dentro del archivo | REVISAR, sin eliminación en la auditoría |
| Q05 | Camas = 0 con ingresos, egresos, pacientes-día o estancias positivos | REVISAR, problema de capacidad |
| Q06 | Pacientes-día > 0 y días-cama disponibles = 0 | ERROR, problema de capacidad |
| Q07 | Camas = pacientes-día positivos × días calendario, métricas enteras y periodo válido | REVISAR, hipótesis de desorden sin corregir columnas |
| Q08 | Pacientes-día / días-cama disponibles > 1,20, con denominador positivo | REVISAR; el tratamiento posterior aparta el grupo en alcance |
| Q09 | Días-cama disponibles / (camas × días calendario) fuera de 0,5–1,5 | Solo REVISAR |

`REVISAR` corresponde a una advertencia de revisión, no a una corrección. Q07 usa
igualdad numérica con tolerancia absoluta 1e-9 y sin tolerancia relativa; un valor
cercano no basta. También se registra Q09 cuando hay cero capacidad teórica y
días-cama positivos, sin dividir por cero. Los límites exactos 0,5 y 1,5 no generan
Q09. No se usa un periodo inválido para calcular Q07/Q09 ni se inventan ceros para
evaluar reglas cruzadas.

`NRO_TOTAL_CAMAS` significa camas. **`NRO_TOTAL_CAMAS_DISPONIB` y
`DIAS_CAMA_DISPONIBLE` significan días-cama disponibles, nunca camas libres.**
Esta semántica de auditoría no cambia las fórmulas existentes de indicadores.

Q07 está motivado por las relaciones comunicadas por el usuario en Dos de Mayo
(2015) y Marino Molina Scippa (2016), como 36 × 31 = 1116. Esa evidencia no permite
inferir un intercambio único de columnas; aquí no se realiza ninguno.

## Interpretación y límites

Una fila puede producir varias reglas: no sumar sus contadores como si fueran
registros distintos. Q01/Q02 cuentan celdas; Q04 cuenta repeticiones posteriores a
la primera. La fila CSV es el índice leído + 2 y puede diferir de la línea física
si existen campos multilínea o líneas vacías. Los códigos IPRESS conservan ceros.

La limpieza histórica sigue deduplicando, convirtiendo números no interpretables
a cero y recortando negativos en memoria; no se rediseña en esta tarea. La nueva
auditoría registra la evidencia antes de esa limpieza. El tratamiento nuevo solo
aparta Q05/Q06/Q07/Q08 en alcance, sin imputar. No constituye una solución general para Q01–Q03.
La etiqueta vigente usa únicamente ocupación observada con umbrales 0.70/0.85;
los percentiles globales y el ratio ya no intervienen en ella. Véase la definición
metodológica en [README.md](../README.md#riesgo-actual). Las diferencias de
indicadores/consolidación con Java permanecen pendientes de una revisión separada.

Las pruebas nuevas usan únicamente CSV artificiales en directorios temporales.
No se ejecuta entrenamiento ni se cambian artefactos `.joblib`.

## Verificación histórica de la implementación de auditoría

Se registró la base antes de modificar el código. El lanzador local no pudo
iniciar Python 3.13, por lo que se usó el intérprete alternativo disponible.
Las dependencias binarias de scikit-learn del entorno del proyecto corresponden
a Python 3.13 y no se pueden importar con ese intérprete alternativo.

| Ejecución | Antes | Después |
| --- | --- | --- |
| Suite completa de pytest | Error al importar `tests/test_entrenar_modelo.py` | Mismo error de dependencia |
| Suite con `--ignore=tests/test_entrenar_modelo.py` | 25 passed, 1 warning | 83 passed, 1 warning |

El warning es `StarletteDeprecationWarning` por el uso de `httpx` en TestClient;
ya aparecía en la ejecución base. No se modificaron dependencias para ocultarlo.
La ejecución posterior incluye tres pruebas del comando `python -m
src.calidad_datos`, con archivos artificiales válidos, Q05 y Q00.

Se compararon 32 archivos mediante SHA-256 (datos, modelos y código protegido)
antes y después de las pruebas, sin cambios. También se verificó mediante AST
que las funciones de consolidación, indicadores, percentiles y target permanecen
iguales. No se regeneraron los datos reales ni se ejecutó entrenamiento.

Queda pendiente ejecutar la suite completa desde el entorno Python compatible
del proyecto. No se afirma que la suite completa haya aprobado.
