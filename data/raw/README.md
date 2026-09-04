# Dataset original

Los archivos D1 usados por el modelo se guardan exclusivamente en la
subcarpeta `Hospitalizacion/`, por ejemplo:

```text
ConsultaD1_Hospitalizaciones_Especialidad_2015_v1.csv
ConsultaD1_Hospitalizaciones_Especialidad_2016_v1.csv
ConsultaD1_Hospitalizaciones_Especialidad_2017_v1.csv
```

El procesamiento no recorre otras subcarpetas de `data/raw`: fuentes como
`Capacidad/` no forman parte de D1. Ordena y concatena todos los `*.csv` de
`Hospitalizacion/`, admite archivos
delimitados por coma o punto y coma y no genera datos sintéticos. Los archivos
vacíos se omiten con una advertencia; si no queda ningún CSV válido, el proceso
termina con un error claro.

`CO_IPRESS` permanece como identificador de texto. El pipeline valida entre
1 y 8 dígitos y lo representa con 8 caracteres usando ceros iniciales; el
literal fuente no se modifica y la metadata registra la transformación.
