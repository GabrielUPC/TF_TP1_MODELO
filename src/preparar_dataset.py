import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .tratamiento_capacidad import apartar_meses_pendientes, huellas_raw, sha256_archivo
    from .datos_raw import (
        ALIAS_COLUMNAS, COLUMNAS_AGRUPACION, COLUMNAS_NUMERICAS, COLUMNAS_TEXTO,
        CSVVacioError, MENSAJE_CSV_VACIO, SECTORES_PUBLICOS, _detectar_formato,
        _es_archivo_temporal, leer_csv, limpiar_texto, listar_archivos_csv,
        mascara_hospitalizacion_valida, mascara_ipress_publicas_lima,
        normalizar_columnas, validar_columnas,
    )
    from .calidad_datos import auditar_directorio
    from .indicadores import agregar_indicadores_dataframe
    from .variables_temporales import (
        COLUMNAS_TEMPORALES,
        agregar_variables_temporales,
    )
else:
    from tratamiento_capacidad import apartar_meses_pendientes, huellas_raw, sha256_archivo
    from datos_raw import (
        ALIAS_COLUMNAS, COLUMNAS_AGRUPACION, COLUMNAS_NUMERICAS, COLUMNAS_TEXTO,
        CSVVacioError, MENSAJE_CSV_VACIO, SECTORES_PUBLICOS, _detectar_formato,
        _es_archivo_temporal, leer_csv, limpiar_texto, listar_archivos_csv,
        mascara_hospitalizacion_valida, mascara_ipress_publicas_lima,
        normalizar_columnas, validar_columnas,
    )
    from calidad_datos import auditar_directorio
    from indicadores import agregar_indicadores_dataframe
    from variables_temporales import (
        COLUMNAS_TEMPORALES,
        agregar_variables_temporales,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "dataset_modelo_ipress.csv"
DATASET_METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"
QUALITY_DIR = PROJECT_ROOT / "data" / "quality"

RENOMBRE_COLUMNAS = {
    "ANHO": "anio",
    "MES": "mes",
    "UBIGEO": "ubigeo",
    "DEPARTAMENTO": "departamento",
    "PROVINCIA": "provincia",
    "DISTRITO": "distrito",
    "SECTOR": "sector",
    "CATEGORIA": "categoria_ipress",
    "CO_IPRESS": "codigo_ipress",
    "RAZON_SOC": "nombre_ipress",
    "ID_HOSPITALIZACION": "id_hospitalizacion",
    "HOSPITALIZACION": "servicio_hospitalizacion",
    "NRO_TOTAL_HOSPIT_ING": "total_ingresos",
    "NRO_TOTAL_HOSPIT_EGR": "total_egresos",
    "NRO_TOTAL_ESTANCIAS": "total_estancias",
    "NRO_TOTAL_PACIENTES_CAMAS": "total_pacientes_camas",
    "NRO_TOTAL_CAMAS": "total_camas",
    "NRO_TOTAL_CAMAS_DISPONIB": "total_camas_disponibles",
    "NRO_TOTAL_FALLECIDOS": "total_fallecidos",
    "ARCHIVO_ORIGEN": "archivo_origen",
}

COLUMNAS_BASE_SALIDA = [
    "anio",
    "mes",
    "periodo_actual",
    "periodo_predicho",
    "ubigeo",
    "departamento",
    "provincia",
    "distrito",
    "sector",
    "categoria_ipress",
    "codigo_ipress",
    "nombre_ipress",
    "id_hospitalizacion",
    "servicio_hospitalizacion",
    "total_ingresos",
    "total_egresos",
    "total_estancias",
    "total_pacientes_camas",
    "total_camas",
    "total_camas_disponibles",
    "total_fallecidos",
    "archivo_origen",
    "dias_mes",
    "promedio_estancia",
    "tasa_fallecidos",
    "ratio_camas_disponibles",
    "ocupacion_estimada",
    "presion_ingresos_camas",
    "rotacion_camas",
    "diferencia_ingresos_egresos",
]

COLUMNAS_OBJETIVO = [
    "nivel_riesgo_actual",
    "nivel_riesgo_actual_codificado",
    "nivel_riesgo_siguiente_mes",
    "nivel_riesgo_siguiente_mes_codificado",
]

COLUMNAS_SALIDA = [
    *COLUMNAS_BASE_SALIDA,
    *COLUMNAS_TEMPORALES,
    *COLUMNAS_OBJETIVO,
]

GRUPO_SERIE_TEMPORAL = [
    "codigo_ipress",
    "servicio_hospitalizacion",
]

MAPEO_RIESGO = {"bajo": 0, "medio": 1, "alto": 2}


def leer_todos_los_csv(
    raw_dir: Path, *, archivos: list[Path] | None = None,
) -> pd.DataFrame:
    archivos = listar_archivos_csv(raw_dir) if archivos is None else archivos

    print(f"Leyendo archivos CSV desde {raw_dir}...")
    datasets: list[pd.DataFrame] = []
    for archivo in archivos:
        try:
            df = leer_csv(archivo)
        except CSVVacioError as error:
            print(f"* ADVERTENCIA: {archivo.name} se omitió: {error}")
            continue

        df = normalizar_columnas(df)
        validar_columnas(df, archivo.name)
        df["ARCHIVO_ORIGEN"] = archivo.name
        datasets.append(df)
        print(f"* {archivo.name}: {len(df)} filas")

    if not datasets:
        raise ValueError(
            "No se encontró ningún CSV con filas y columnas válidas en "
            f"{raw_dir}."
        )

    combinado = pd.concat(datasets, ignore_index=True, sort=False)
    print(f"Total combinado: {len(combinado)} filas")
    return combinado


def limpiar_registros(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.drop_duplicates().copy()
    for columna in COLUMNAS_TEXTO:
        resultado[columna] = limpiar_texto(resultado[columna])

    resultado = resultado.loc[mascara_hospitalizacion_valida(resultado)].copy()
    resultado["ANHO"] = pd.to_numeric(resultado["ANHO"], errors="coerce")
    resultado["MES"] = pd.to_numeric(resultado["MES"], errors="coerce")
    resultado = resultado.dropna(subset=["ANHO", "MES"])
    resultado = resultado[resultado["MES"].between(1, 12)]
    resultado[["ANHO", "MES"]] = resultado[["ANHO", "MES"]].astype(int)

    for columna in COLUMNAS_NUMERICAS:
        valores_limpios = (
            resultado[columna]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        resultado[columna] = (
            pd.to_numeric(valores_limpios, errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )
    return resultado


def filtrar_ipress_publicas_lima(df: pd.DataFrame) -> pd.DataFrame:
    mascara = mascara_ipress_publicas_lima(df)
    resultado = df.loc[mascara].copy()
    if resultado.empty:
        raise ValueError(
            "No quedaron registros de IPRESS públicas de Lima Metropolitana "
            "después de aplicar los filtros."
        )
    return resultado


def consolidar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(COLUMNAS_AGRUPACION, as_index=False, dropna=False)[
            COLUMNAS_NUMERICAS
        ]
        .sum()
        .reset_index(drop=True)
    )


def crear_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.rename(columns=RENOMBRE_COLUMNAS)
    return agregar_indicadores_dataframe(resultado)


def crear_riesgo_actual(
    df: pd.DataFrame,
    percentiles: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Etiqueta operacional basada exclusivamente en ocupación observada.

    Se conserva la firma y la tupla de retorno por compatibilidad. El argumento
    legado percentiles se ignora y el segundo resultado es siempre {}:
    ya no se calculan ni se usan percentiles para construir la etiqueta.
    """
    resultado = df.copy()
    ocupacion = pd.to_numeric(resultado["ocupacion_estimada"], errors="raise")
    if not (np.isfinite(ocupacion) & ocupacion.ge(0)).all():
        raise ValueError("La ocupación para etiquetar debe ser finita y no negativa.")
    resultado["nivel_riesgo_actual"] = np.select(
        [ocupacion.ge(0.85), ocupacion.ge(0.70)],
        ["alto", "medio"],
        default="bajo",
    )
    resultado["nivel_riesgo_actual_codificado"] = (
        resultado["nivel_riesgo_actual"].map(MAPEO_RIESGO).astype(int)
    )
    return resultado, {}


def crear_objetivo_futuro(df: pd.DataFrame) -> pd.DataFrame:
    requeridas = {
        *GRUPO_SERIE_TEMPORAL,
        "anio",
        "mes",
        "nivel_riesgo_actual",
        "nivel_riesgo_actual_codificado",
    }
    faltantes = sorted(requeridas.difference(df.columns))
    if faltantes:
        raise ValueError(
            "Faltan columnas para crear el objetivo futuro: "
            + ", ".join(faltantes)
        )

    resultado = df.copy()
    resultado["_orden_original"] = np.arange(len(resultado))
    resultado["_periodo"] = pd.to_datetime(
        {
            "year": resultado["anio"].astype(int),
            "month": resultado["mes"].astype(int),
            "day": 1,
        }
    ).dt.to_period("M")
    resultado = resultado.sort_values(
        [*GRUPO_SERIE_TEMPORAL, "_periodo", "_orden_original"]
    )

    duplicados = resultado.duplicated(
        [*GRUPO_SERIE_TEMPORAL, "_periodo"],
        keep=False,
    )
    if duplicados.any():
        raise ValueError(
            "Existen periodos duplicados por IPRESS y servicio; no es seguro "
            "crear una etiqueta del mes siguiente."
        )

    grupos = resultado.groupby(
        GRUPO_SERIE_TEMPORAL,
        sort=False,
        observed=True,
    )
    riesgo_siguiente = grupos["nivel_riesgo_actual"].shift(-1)
    codigo_siguiente = grupos["nivel_riesgo_actual_codificado"].shift(-1)
    periodo_observado_siguiente = grupos["_periodo"].shift(-1)
    periodo_esperado = resultado["_periodo"] + 1
    tiene_mes_siguiente = periodo_observado_siguiente.eq(periodo_esperado)

    resultado["periodo_actual"] = resultado["_periodo"].astype(str)
    resultado["periodo_predicho"] = periodo_esperado.astype(str)
    resultado["nivel_riesgo_siguiente_mes"] = riesgo_siguiente.where(
        tiene_mes_siguiente
    )
    resultado["nivel_riesgo_siguiente_mes_codificado"] = (
        codigo_siguiente.where(tiene_mes_siguiente)
    )

    resultado = resultado.loc[tiene_mes_siguiente].copy()
    resultado["nivel_riesgo_siguiente_mes_codificado"] = resultado[
        "nivel_riesgo_siguiente_mes_codificado"
    ].astype(int)
    return (
        resultado.sort_values("_orden_original")
        .drop(columns=["_orden_original", "_periodo"])
        .reset_index(drop=True)
    )


def crear_variable_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibilidad: crea riesgo actual y objetivo del mes siguiente."""
    con_riesgo, _ = crear_riesgo_actual(df)
    return crear_objetivo_futuro(con_riesgo)


def mostrar_validaciones_indicadores(df: pd.DataFrame) -> None:
    indicadores = [
        "ratio_camas_disponibles",
        "ocupacion_estimada",
        "presion_ingresos_camas",
    ]
    print("\nEstadísticos descriptivos de los indicadores:")
    print(df[indicadores].describe().transpose())

    ratios_atipicos = df["ratio_camas_disponibles"].gt(1.5)
    if ratios_atipicos.any():
        print(
            "\nADVERTENCIA: se encontraron "
            f"{int(ratios_atipicos.sum())} registros con "
            "ratio_camas_disponibles mayor a 1.5."
        )

    presion_sin_camas = df["total_camas"].eq(0) & df["total_ingresos"].gt(0)
    if presion_sin_camas.any():
        print(
            "\nADVERTENCIA: se encontraron "
            f"{int(presion_sin_camas.sum())} registros con ingresos y cero "
            "camas. La presión se conserva igual a los ingresos como regla "
            "controlada de presión extrema."
        )


def _guardar_metadata_dataset(
    df_antes_objetivo: pd.DataFrame,
    df_entrenamiento: pd.DataFrame,
    percentiles: dict[str, float],
    tratamiento: dict,
    fuentes: dict[str, str],
) -> None:
    archivos = sorted(
        df_antes_objetivo["archivo_origen"].dropna().astype(str).unique().tolist()
    )
    periodos = sorted(
        df_antes_objetivo.assign(
            periodo=df_antes_objetivo["anio"].astype(str)
            + "-"
            + df_antes_objetivo["mes"].astype(str).str.zfill(2)
        )["periodo"].unique().tolist()
    )
    metadata = {
        "tratamiento_capacidad": tratamiento,
        "dataset_sha256": sha256_archivo(OUTPUT_PATH),
        "raw_sha256": fuentes,
        "numero_filas_antes_objetivo_futuro": int(df_antes_objetivo.shape[0]),
        "numero_filas_entrenamiento": int(df_entrenamiento.shape[0]),
        "filas_eliminadas_sin_mes_siguiente": int(
            df_antes_objetivo.shape[0] - df_entrenamiento.shape[0]
        ),
        "anios_disponibles": sorted(
            df_antes_objetivo["anio"].astype(int).unique().tolist()
        ),
        "periodos_disponibles": periodos,
        "cantidad_archivos_raw_leidos": len(archivos),
        "archivos_raw_leidos": archivos,
        "percentiles_riesgo_actual": {},
        "metodo_percentiles": "No se utilizan percentiles para construir la etiqueta.",
        "definicion_target": {
            "version": "riesgo_ocupacion_observada_v2",
            "indicador": "ocupacion_estimada",
            "formula": "total_pacientes_camas / total_camas_disponibles",
            "unidad": "pacientes-día / días-cama disponibles",
            "nivel_riesgo_actual": {
                "bajo": "ocupacion_estimada < 0.70",
                "medio": "0.70 <= ocupacion_estimada < 0.85",
                "alto": "ocupacion_estimada >= 0.85",
            },
            "codificacion": MAPEO_RIESGO,
            "nivel_riesgo_siguiente_mes": (
                "nivel_riesgo_actual del mes calendario t+1 de la misma IPRESS y servicio"
            ),
            "usa_ratio_camas_disponibles": False,
            "usa_percentiles_presion_ingresos_camas": False,
        },
        "regla_continuidad_objetivo": (
            "Solo se conserva una fila cuando existe exactamente el siguiente "
            "mes calendario para la misma IPRESS y servicio."
        ),
    }
    DATASET_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def preparar_dataset() -> pd.DataFrame:
    fuentes = huellas_raw(RAW_DATA_DIR)
    # Los reportes deben existir antes de toda limpieza destructiva.
    auditoria = auditar_directorio(RAW_DATA_DIR, QUALITY_DIR)
    df = leer_todos_los_csv(RAW_DATA_DIR, archivos=auditoria.archivos_validos)
    filas_combinadas = len(df)

    print("\nLimpiando registros...")
    df = limpiar_registros(df)
    filas_limpias = len(df)

    print("Filtrando IPRESS públicas de Lima Metropolitana...")
    df = filtrar_ipress_publicas_lima(df)
    filas_filtradas = len(df)

    print("Apartando meses pendientes Q05/Q06/Q07/Q08 antes de consolidar e inferir riesgo...")
    df, tratamiento = apartar_meses_pendientes(df, QUALITY_DIR, raw_sha256=fuentes)

    print("Consolidando datos mensuales por IPRESS, servicio y origen...")
    df = consolidar_dataset(df)

    print("Calculando indicadores y variables temporales...")
    df = crear_indicadores(df)
    df = agregar_variables_temporales(df)
    df, percentiles = crear_riesgo_actual(df)
    filas_antes_objetivo = len(df)

    print("Creando objetivo de riesgo para el mes siguiente...")
    df_entrenamiento = crear_objetivo_futuro(df)
    filas_eliminadas = filas_antes_objetivo - len(df_entrenamiento)
    df_entrenamiento = df_entrenamiento[COLUMNAS_SALIDA]

    if df_entrenamiento.empty:
        raise ValueError("No quedan parejas de meses consecutivos después del tratamiento de capacidad.")
    if huellas_raw(RAW_DATA_DIR) != fuentes:
        raise ValueError("Los RAW cambiaron durante la preparación; vuelva a ejecutar sin modificar las fuentes.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_entrenamiento.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    _guardar_metadata_dataset(df, df_entrenamiento, percentiles, tratamiento, fuentes)

    periodos = sorted(df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2))
    print("\nDataset preparado correctamente.")
    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Metadata generada: {DATASET_METADATA_PATH}")
    print(f"Filas combinadas: {filas_combinadas}")
    print(f"Filas después de limpieza: {filas_limpias}")
    print(f"Filas después del filtro Lima pública: {filas_filtradas}")
    print(f"Filas finales antes del objetivo futuro: {filas_antes_objetivo}")
    print(f"Filas eliminadas por no tener mes siguiente: {filas_eliminadas}")
    print(f"Filas finales de entrenamiento: {len(df_entrenamiento)}")
    print(f"Años detectados: {sorted(df['anio'].unique().tolist())}")
    print(
        "Periodos detectados: "
        f"{periodos[0]} a {periodos[-1]} ({len(set(periodos))} periodos)"
    )
    mostrar_validaciones_indicadores(df)
    print("\nDistribución de nivel_riesgo_actual:")
    print(df["nivel_riesgo_actual"].value_counts(dropna=False))
    print("\nDistribución de nivel_riesgo_siguiente_mes:")
    print(
        df_entrenamiento["nivel_riesgo_siguiente_mes"].value_counts(
            dropna=False
        )
    )
    print("\nDistribución de nivel_riesgo_siguiente_mes_codificado:")
    print(
        df_entrenamiento[
            "nivel_riesgo_siguiente_mes_codificado"
        ].value_counts(dropna=False).sort_index()
    )
    return df_entrenamiento


if __name__ == "__main__":
    try:
        preparar_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"Error al preparar el dataset: {error}") from error
