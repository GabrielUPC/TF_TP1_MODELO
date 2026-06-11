# Dataset original

Coloque en esta carpeta uno o más archivos CSV, por ejemplo:

```text
ConsultaD1_Hospitalizaciones_Especialidad_2015_v1.csv
ConsultaD1_Hospitalizaciones_Especialidad_2016_v1.csv
ConsultaD1_Hospitalizaciones_Especialidad_2017_v1.csv
```

El procesamiento ordena y concatena todos los `*.csv`, admite archivos
delimitados por coma o punto y coma y no genera datos sintéticos. Los archivos
vacíos se omiten con una advertencia; si no queda ningún CSV válido, el proceso
termina con un error claro.
