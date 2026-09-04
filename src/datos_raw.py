"""Lectura y convenciones RAW compartidas; no modifica archivos de origen."""
import codecs
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT_DIR = PROJECT_ROOT / "data" / "raw"
# Directorio dedicado exclusivamente a los D1 de hospitalización. No se hace
# búsqueda recursiva: Capacidad y futuras fuentes RAW no deben entrar por azar.
RAW_DATA_DIR = RAW_ROOT_DIR / "Hospitalizacion"
LONGITUD_CODIGO_IPRESS = 8
PREFIJOS_SERVICIOS_MODELO = ("24", "25")

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
    "ARCHIVO_ORIGEN",
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

ALIAS_COLUMNAS = {
    "DIAS_CAMA_DISPONIBLE": "NRO_TOTAL_CAMAS_DISPONIB",
}

MENSAJE_CSV_VACIO = (
    "El archivo CSV original está vacío. Coloque datasets reales en data/raw/ "
    "antes de ejecutar el procesamiento."
)


class CSVVacioError(ValueError):
    """Indica que un CSV existe pero no aporta filas utilizables."""


def _detectar_formato(path: Path) -> tuple[str, str]:
    with path.open("rb") as entrada:
        muestra = entrada.read(8192)
    for encoding in ("utf-8-sig", "latin1"):
        try:
            texto = codecs.getincrementaldecoder(encoding)().decode(muestra, final=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"No se pudo determinar la codificación de {path.name}.")

    primera_linea = next(
        (linea for linea in texto.splitlines() if linea.strip()),
        "",
    )
    if not primera_linea:
        raise CSVVacioError(MENSAJE_CSV_VACIO)

    separador = (
        ";" if primera_linea.count(";") > primera_linea.count(",") else ","
    )
    return encoding, separador


def leer_csv(path: Path, *, conservar_originales: bool = False) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {path}")
    if path.stat().st_size == 0:
        raise CSVVacioError(MENSAJE_CSV_VACIO)

    encoding, separador = _detectar_formato(path)
    try:
        df = pd.read_csv(
            path,
            encoding=encoding,
            sep=separador,
            dtype=str,
            keep_default_na=not conservar_originales,
        )
    except pd.errors.EmptyDataError as error:
        raise CSVVacioError(MENSAJE_CSV_VACIO) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"No se pudo interpretar el archivo {path.name}: {error}"
        ) from error

    if df.columns.empty:
        raise CSVVacioError(f"El archivo {path.name} no contiene columnas.")
    if df.empty:
        raise CSVVacioError(f"El archivo {path.name} no contiene filas.")
    return df


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    resultado.columns = [
        re.sub(r"\s+", "_", str(columna).strip().upper())
        for columna in resultado.columns
    ]
    renombres = {
        alias: canonica
        for alias, canonica in ALIAS_COLUMNAS.items()
        if alias in resultado.columns and canonica not in resultado.columns
    }
    return resultado.rename(columns=renombres)


def validar_columnas(df: pd.DataFrame, archivo: str | None = None) -> None:
    requeridas = set(COLUMNAS_AGRUPACION[:-1] + COLUMNAS_NUMERICAS)
    faltantes = sorted(requeridas.difference(df.columns))
    if faltantes:
        origen = f" en el archivo {archivo}" if archivo else ""
        raise ValueError(
            f"Faltan columnas obligatorias{origen}: " + ", ".join(faltantes)
        )


def _es_archivo_temporal(path: Path) -> bool:
    return path.name.startswith(("~", ".", "$"))


def limpiar_texto(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def normalizar_codigo_ipress(valor):
    """Devuelve el código RENIPRESS canónico de 8 caracteres o ``pd.NA``.

    El código es un identificador, nunca una variable continua. Se admiten
    entre 1 y 8 dígitos y el sufijo ``.0`` de fuentes tabulares históricas; no
    se aceptan decimales reales, signos, texto ni valores de más de 8 dígitos.
    """
    if pd.isna(valor):
        return pd.NA
    texto = str(valor).strip()
    if not re.fullmatch(r"[0-9]{1,8}(?:\.0+)?", texto):
        return pd.NA
    return texto.split(".")[0].zfill(LONGITUD_CODIGO_IPRESS)


def normalizar_codigos_ipress(
    serie: pd.Series, *, rechazar_invalidos: bool = True,
) -> pd.Series:
    """Normaliza una serie sin alterar el argumento y controla inválidos."""
    originales = serie.astype("string")
    resultado = originales.map(normalizar_codigo_ipress).astype("string")
    invalidos = resultado.isna()
    if rechazar_invalidos and invalidos.any():
        ejemplos = originales.loc[invalidos].head(5).tolist()
        raise ValueError(
            "CO_IPRESS debe contener entre 1 y 8 dígitos; "
            f"valores inválidos de ejemplo: {ejemplos}."
        )
    return resultado


def resumir_normalizacion_codigo_ipress(serie: pd.Series) -> dict:
    """Trazabilidad agregada; el literal original permanece en el RAW."""
    originales = serie.astype("string").str.strip()
    canonicos = normalizar_codigos_ipress(originales)
    cambiados = originales.ne(canonicos)
    ejemplos = (
        pd.DataFrame({"original": originales, "canonico": canonicos})
        .loc[cambiados]
        .drop_duplicates()
        .head(20)
        .to_dict(orient="records")
    )
    return {
        "version": "renipress_8_caracteres_v1",
        "tipo": "identificador_string",
        "longitud_canonica": LONGITUD_CODIGO_IPRESS,
        "regla": "strip -> validar 1-8 dígitos (sufijo .0 permitido) -> zfill(8)",
        "filas_evaluadas": int(len(originales)),
        "filas_transformadas": int(cambiados.sum()),
        "codigos_distintos_originales": int(originales.nunique()),
        "codigos_distintos_canonicos": int(canonicos.nunique()),
        "ejemplos": ejemplos,
        "trazabilidad_original": "El literal original permanece inmutable en data/raw y se identifica mediante archivo_origen.",
    }


def listar_archivos_csv(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No se encontró el directorio raw: {raw_dir}")
    archivos = sorted(
        (p for p in raw_dir.glob("*.csv") if p.is_file() and not _es_archivo_temporal(p)),
        key=lambda p: p.name.lower(),
    )
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos CSV válidos en {raw_dir}.")
    return archivos


def mascara_ipress_publicas_lima(df: pd.DataFrame) -> pd.Series:
    """Recibe textos normalizados, igual que el filtro histórico del pipeline."""
    return (
        df["DEPARTAMENTO"].eq("LIMA")
        & df["PROVINCIA"].eq("LIMA")
        & df["SECTOR"].isin(SECTORES_PUBLICOS)
    )


def mascara_hospitalizacion_valida(df: pd.DataFrame) -> pd.Series:
    """Incluye hospitalización (24) y cuidados críticos (25) por ID.

    El ID se compara como texto, sin conversión numérica ni pérdida de ceros.
    El nombre del servicio no determina el alcance; incluye 245600 (de día).
    """
    ids = df["ID_HOSPITALIZACION"].astype("string").str.strip()
    return ids.str.startswith(PREFIJOS_SERVICIOS_MODELO, na=False)
