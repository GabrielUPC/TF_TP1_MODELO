from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


CLAVE_GRANULARIDAD = [
    "codigo_ipress",
    "anio",
    "mes",
    "servicio_hospitalizacion",
]

ALIAS_COLUMNAS = {
    "codigo_renipress": "codigo_ipress",
    "servicio_hospitalario": "servicio_hospitalizacion",
    "ingresos": "total_ingresos",
    "egresos": "total_egresos",
    "estancias": "total_estancias",
    "pacientes_cama": "total_pacientes_camas",
    "camas_totales": "total_camas",
    "camas_disponibles_habilitadas": "total_camas_disponibles",
}

COLUMNAS_REQUERIDAS = [
    "codigo_ipress",
    "anio",
    "mes",
    "servicio_hospitalizacion",
    "total_ingresos",
    "total_egresos",
    "total_estancias",
    "total_pacientes_camas",
    "total_camas",
    "total_camas_disponibles",
]

COLUMNAS_NUMERICAS = [
    "total_ingresos",
    "total_egresos",
    "total_estancias",
    "total_pacientes_camas",
    "total_camas",
    "total_camas_disponibles",
]

MENSAJE_DUPLICADO = (
    "No se permite más de un registro vigente con la misma IPRESS, año, mes "
    "y servicio hospitalario."
)

MENSAJE_GRANULARIDAD = (
    "La información se interpreta como mensual, no diaria, no por paciente "
    "y no en tiempo real."
)


def normalizar_nombre_columna(nombre: Any) -> str:
    texto = str(nombre).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(caracter for caracter in texto if not unicodedata.combining(caracter))
    for origen, destino in {
        " ": "_",
        "-": "_",
        ".": "_",
        "/": "_",
        "\\": "_",
    }.items():
        texto = texto.replace(origen, destino)
    while "__" in texto:
        texto = texto.replace("__", "_")
    return texto.strip("_")


def leer_archivo(ruta: Path) -> pd.DataFrame:
    extension = ruta.suffix.lower()
    if extension == ".csv":
        errores = []
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            try:
                return pd.read_csv(
                    ruta,
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    dtype=str,
                )
            except UnicodeDecodeError as error:
                errores.append(str(error))
        raise ValueError("No se pudo leer el CSV con utf-8-sig, utf-8 ni latin1.")

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(ruta, dtype=str)

    raise ValueError("El archivo debe tener extensión .csv, .xlsx o .xls.")


def normalizar_columnas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    advertencias: list[str] = []
    columnas = [normalizar_nombre_columna(columna) for columna in df.columns]
    normalizadas = [ALIAS_COLUMNAS.get(columna, columna) for columna in columnas]

    vistos: set[str] = set()
    columnas_finales: list[str] = []
    indices_conservar: list[int] = []
    for indice, columna in enumerate(normalizadas):
        if columna in vistos:
            advertencias.append(
                f"Columna duplicada tras normalización omitida: {columna}"
            )
            continue
        vistos.add(columna)
        columnas_finales.append(columna)
        indices_conservar.append(indice)

    resultado = df.iloc[:, indices_conservar].copy()
    resultado.columns = columnas_finales
    return resultado, advertencias


def es_vacio(valor: Any) -> bool:
    return pd.isna(valor) or str(valor).strip() == ""


def validar(df: pd.DataFrame) -> dict[str, Any]:
    errores: list[str] = []
    advertencias: list[str] = []

    df, advertencias_columnas = normalizar_columnas(df)
    advertencias.extend(advertencias_columnas)

    faltantes = [columna for columna in COLUMNAS_REQUERIDAS if columna not in df.columns]
    if faltantes:
        errores.append("Faltan columnas obligatorias: " + ", ".join(faltantes))
        return {
            "dataframe": df,
            "errores": errores,
            "advertencias": advertencias,
            "duplicados": pd.DataFrame(),
            "registros_validos": 0,
        }

    filas_invalidas: set[int] = set()
    for indice, fila in df.iterrows():
        numero_fila = int(indice) + 2
        for columna in COLUMNAS_REQUERIDAS:
            if es_vacio(fila[columna]):
                errores.append(
                    f"Fila {numero_fila}: el campo {columna} es obligatorio."
                )
                filas_invalidas.add(int(indice))

    anios = pd.to_numeric(df["anio"], errors="coerce")
    meses = pd.to_numeric(df["mes"], errors="coerce")
    for indice, valor in anios.items():
        if pd.isna(valor) or int(valor) < 1900 or int(valor) > 2100:
            errores.append(f"Fila {int(indice) + 2}: año inválido.")
            filas_invalidas.add(int(indice))
    for indice, valor in meses.items():
        if pd.isna(valor) or int(valor) < 1 or int(valor) > 12:
            errores.append(f"Fila {int(indice) + 2}: mes inválido.")
            filas_invalidas.add(int(indice))

    for columna in COLUMNAS_NUMERICAS:
        valores = pd.to_numeric(df[columna], errors="coerce")
        for indice, valor in valores.items():
            if pd.isna(valor) or float(valor) < 0:
                errores.append(
                    f"Fila {int(indice) + 2}: {columna} debe ser numérico no negativo."
                )
                filas_invalidas.add(int(indice))

    claves = df[CLAVE_GRANULARIDAD].copy()
    claves["codigo_ipress"] = claves["codigo_ipress"].astype(str).str.strip()
    claves["servicio_hospitalizacion"] = (
        claves["servicio_hospitalizacion"].astype(str).str.strip().str.upper()
    )
    claves["anio"] = pd.to_numeric(claves["anio"], errors="coerce").astype("Int64")
    claves["mes"] = pd.to_numeric(claves["mes"], errors="coerce").astype("Int64")

    duplicados = df[claves.duplicated(keep=False)].copy()
    if not duplicados.empty:
        errores.append(MENSAJE_DUPLICADO)
        filas_invalidas.update(int(indice) for indice in duplicados.index)

    return {
        "dataframe": df,
        "errores": errores,
        "advertencias": advertencias,
        "duplicados": duplicados,
        "registros_validos": max(len(df) - len(filas_invalidas), 0),
    }


def imprimir_resultado(ruta: Path, resultado: dict[str, Any]) -> None:
    df = resultado["dataframe"]
    errores = resultado["errores"]
    advertencias = resultado["advertencias"]
    duplicados = resultado["duplicados"]

    print(f"Archivo validado: {ruta}")
    print(f"Registros leídos: {len(df)}")
    print(f"Registros válidos: {resultado['registros_validos']}")
    print(f"Errores: {len(errores)}")
    for error in errores:
        print(f"  - {error}")
    print(f"Advertencias: {len(advertencias)}")
    for advertencia in advertencias:
        print(f"  - {advertencia}")
    print(f"Duplicados encontrados: {len(duplicados)}")
    if not duplicados.empty:
        print(
            duplicados[CLAVE_GRANULARIDAD]
            .drop_duplicates()
            .to_string(index=False)
        )
    print("Clave de granularidad usada: " + " + ".join(CLAVE_GRANULARIDAD))
    print(MENSAJE_GRANULARIDAD)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida plantilla mensual de hospitalización y camas."
    )
    parser.add_argument("ruta", help="Ruta del archivo CSV o Excel a validar.")
    args = parser.parse_args()

    ruta = Path(args.ruta)
    if not ruta.is_file():
        print(f"No existe el archivo: {ruta}", file=sys.stderr)
        return 2

    try:
        df = leer_archivo(ruta)
        resultado = validar(df)
        imprimir_resultado(ruta, resultado)
        return 1 if resultado["errores"] else 0
    except Exception as error:
        print(f"Error al validar plantilla: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
