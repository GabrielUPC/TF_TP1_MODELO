from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ConsultaD1_Hospitalizaciones_Especialidad_2015_v1.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "dataset_modelo_ipress.csv"

SECTORES_PUBLICOS = {
    "MINSA",
    "ESSALUD",
    "GOBIERNO REGIONAL",
    "SANIDAD DE LA POLICIA NACIONAL DEL PERU",
    "SANIDAD DEL EJERCITO DEL PERU",
    "SANIDAD DE LA MARINA DE GUERRA DEL PERU",
    "SANIDAD DE LA FUERZA AEREA DEL PERU",
}

COLUMNAS_NUMERICAS = [
    "NRO_TOTAL_HOSPIT_ING",
    "NRO_TOTAL_HOSPIT_EGR",
    "NRO_TOTAL_ESTANCIAS",
    "NRO_TOTAL_PACIENTES_CAMAS",
    "NRO_TOTAL_CAMAS",
    "NRO_TOTAL_CAMAS_DISPONIB",
    "NRO_TOTAL_FALLECIDOS",
]

COLUMNAS_AGRUPACION = [
    "ANHO",
    "MES",
    "UBIGEO",
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
    "SECTOR",
    "CATEGORIA",
    "CO_IPRESS",
    "RAZON_SOC",
    "ID_HOSPITALIZACION",
    "HOSPITALIZACION",
]

COLUMNAS_TEXTO = [
    "UBIGEO",
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
    "SECTOR",
    "CATEGORIA",
    "CO_IPRESS",
    "RAZON_SOC",
    "ID_HOSPITALIZACION",
    "HOSPITALIZACION",
]

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
}


def leer_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {path}")

    for encoding in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"No se pudo determinar la codificación del archivo: {path}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    resultado.columns = [
        re.sub(r"\s+", "_", str(columna).strip().upper())
        for columna in resultado.columns
    ]
    return resultado


def validar_columnas(df: pd.DataFrame) -> None:
    requeridas = set(COLUMNAS_AGRUPACION + COLUMNAS_NUMERICAS)
    faltantes = sorted(requeridas.difference(df.columns))
    if faltantes:
        raise ValueError(
            "El archivo fuente no contiene las columnas requeridas: "
            + ", ".join(faltantes)
        )


def limpiar_texto(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def limpiar_registros(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.drop_duplicates().copy()

    for columna in COLUMNAS_TEXTO:
        resultado[columna] = limpiar_texto(resultado[columna])

    resultado = resultado[resultado["HOSPITALIZACION"].ne("")]
    resultado = resultado[
        ~resultado["ID_HOSPITALIZACION"].isin({"NE_0001", "NE_0002"})
    ]

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
    mascara = (
        df["DEPARTAMENTO"].eq("LIMA")
        & df["PROVINCIA"].eq("LIMA")
        & df["SECTOR"].isin(SECTORES_PUBLICOS)
    )
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
    resultado = df.copy()
    resultado["dias_mes"] = pd.to_datetime(
        {
            "year": resultado["ANHO"],
            "month": resultado["MES"],
            "day": 1,
        }
    ).dt.days_in_month

    egresos = resultado["NRO_TOTAL_HOSPIT_EGR"]
    camas = resultado["NRO_TOTAL_CAMAS"]
    camas_disponibles = resultado["NRO_TOTAL_CAMAS_DISPONIB"]
    capacidad_mensual = camas * resultado["dias_mes"]

    resultado["promedio_estancia"] = np.where(
        egresos > 0,
        resultado["NRO_TOTAL_ESTANCIAS"] / egresos,
        0.0,
    )
    resultado["tasa_fallecidos"] = np.where(
        egresos > 0,
        resultado["NRO_TOTAL_FALLECIDOS"] / egresos,
        0.0,
    )
    resultado["ratio_camas_disponibles"] = np.where(
        camas > 0,
        camas_disponibles / camas,
        0.0,
    )
    resultado["ocupacion_estimada"] = np.where(
        capacidad_mensual > 0,
        resultado["NRO_TOTAL_PACIENTES_CAMAS"] / capacidad_mensual,
        0.0,
    )
    resultado["presion_ingresos_camas"] = np.where(
        camas_disponibles > 0,
        resultado["NRO_TOTAL_HOSPIT_ING"] / camas_disponibles,
        0.0,
    )
    resultado["rotacion_camas"] = np.where(
        camas > 0,
        egresos / camas,
        0.0,
    )
    resultado["diferencia_ingresos_egresos"] = (
        resultado["NRO_TOTAL_HOSPIT_ING"] - egresos
    )

    return resultado


def crear_variable_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    presion = resultado["presion_ingresos_camas"]
    percentil_50 = presion.quantile(0.50)
    percentil_75 = presion.quantile(0.75)

    riesgo_alto = (
        resultado["ocupacion_estimada"].ge(0.85)
        | resultado["ratio_camas_disponibles"].le(0.10)
        | presion.ge(percentil_75)
    )
    riesgo_medio = (
        resultado["ocupacion_estimada"].ge(0.70)
        | resultado["ratio_camas_disponibles"].le(0.20)
        | presion.ge(percentil_50)
    )

    resultado["nivel_riesgo"] = np.select(
        [riesgo_alto, riesgo_medio],
        ["alto", "medio"],
        default="bajo",
    )
    resultado["nivel_riesgo_codificado"] = (
        resultado["nivel_riesgo"]
        .map({"bajo": 0, "medio": 1, "alto": 2})
        .astype(int)
    )
    return resultado


def preparar_dataset() -> pd.DataFrame:
    print("Leyendo dataset original...")
    df = leer_csv(RAW_DATA_PATH)

    print("Normalizando y validando columnas...")
    df = normalizar_columnas(df)
    validar_columnas(df)

    print("Limpiando y filtrando registros...")
    df = limpiar_registros(df)
    df = filtrar_ipress_publicas_lima(df)

    print("Consolidando datos mensuales por IPRESS y servicio...")
    df = consolidar_dataset(df)

    print("Calculando indicadores y variable objetivo...")
    df = crear_indicadores(df)
    df = crear_variable_objetivo(df)
    df = df.rename(columns=RENOMBRE_COLUMNAS)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nDataset preparado correctamente.")
    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Número de filas: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    print(f"Columnas finales: {df.columns.tolist()}")
    print("\nDistribución de categoria_ipress:")
    print(df["categoria_ipress"].value_counts(dropna=False))
    print("\nDistribución de nivel_riesgo:")
    print(df["nivel_riesgo"].value_counts(dropna=False))
    print("\nDistribución de nivel_riesgo_codificado:")
    print(df["nivel_riesgo_codificado"].value_counts(dropna=False).sort_index())

    return df


if __name__ == "__main__":
    try:
        preparar_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"Error al preparar el dataset: {error}") from error
