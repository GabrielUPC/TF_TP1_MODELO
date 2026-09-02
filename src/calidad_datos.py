"""Auditoría Q00-Q09 sobre valores RAW, sin limpiar ni modificar el origen."""
import argparse
import json
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .datos_raw import (
        ALIAS_COLUMNAS, COLUMNAS_NUMERICAS, PROJECT_ROOT, RAW_DATA_DIR,
        _detectar_formato, leer_csv, limpiar_texto, listar_archivos_csv,
        mascara_hospitalizacion_valida, mascara_ipress_publicas_lima,
        normalizar_columnas, validar_columnas,
    )
else:
    from datos_raw import (
        ALIAS_COLUMNAS, COLUMNAS_NUMERICAS, PROJECT_ROOT, RAW_DATA_DIR,
        _detectar_formato, leer_csv, limpiar_texto, listar_archivos_csv,
        mascara_hospitalizacion_valida, mascara_ipress_publicas_lima,
        normalizar_columnas, validar_columnas,
    )

QUALITY_DIR = PROJECT_ROOT / "data" / "quality"
ANIO_MINIMO = 1900
ANIO_MAXIMO = 2100
UMBRAL_OCUPACION = 1.20
RELACION_DIAS_CAMA_MINIMA = 0.50
RELACION_DIAS_CAMA_MAXIMA = 1.50
DIAS_CAMA = "NRO_TOTAL_CAMAS_DISPONIB"  # Días-cama, nunca camas libres.
CAMAS = "NRO_TOTAL_CAMAS"
PACIENTES_DIA = "NRO_TOTAL_PACIENTES_CAMAS"
REGLAS = {
    "Q00": ("errores_esquema", "ERROR", "Esquema o lectura no utilizable"),
    "Q01": ("valores_numericos_invalidos", "ERROR", "Valor numérico ausente, no convertible o no finito"),
    "Q02": ("valores_negativos", "ERROR", "Valor negativo"),
    "Q03": ("periodos_invalidos", "ERROR", "Año entero 1900-2100 y mes entero 1-12 requeridos"),
    "Q04": ("duplicados_exactos", "REVISAR", "Fila repetida exactamente dentro del archivo; se conserva"),
    "Q05": ("cero_camas_con_actividad", "REVISAR", "Cero camas con actividad hospitalaria"),
    "Q06": ("pacientes_dia_sin_dias_cama", "ERROR", "Pacientes-día positivos con cero días-cama disponibles"),
    "Q07": ("posibles_desplazamientos", "REVISAR", "Camas = pacientes-día × días del mes: posible desplazamiento, no confirmado"),
    "Q08": ("ocupacion_mayor_120pct", "REVISAR", "Ocupación auditada mayor a 120%"),
    "Q09": ("relacion_dias_cama_extrema", "REVISAR", "Desviación fuerte entre días-cama disponibles y camas × días calendario"),
}
COLUMNAS_HALLAZGOS = [
    "archivo", "fila_csv_aproximada", "regla", "severidad", "descripcion",
    "anio", "mes", "codigo_ipress", "nombre_ipress", "servicio_hospitalizacion",
    "valores_relevantes", "en_alcance_modelo",
]
CONTEXTOS = {
    "anio": "ANHO", "mes": "MES", "codigo_ipress": "CO_IPRESS",
    "nombre_ipress": "RAZON_SOC", "servicio_hospitalizacion": "HOSPITALIZACION",
}
NOMBRES_REPORTES = ("resumen_calidad.csv", "hallazgos_calidad.csv", "resumen_calidad.json")


@dataclass
class ResultadoArchivo:
    resumen: dict
    hallazgos: pd.DataFrame


@dataclass
class ResultadoAuditoria:
    archivos_validos: list[Path]
    resumen: dict


def _valor_json(valor):
    if pd.isna(valor):
        return None
    if isinstance(valor, (float, np.floating)) and not np.isfinite(valor):
        return str(valor)
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def _numeros(serie: pd.Series, *, quitar_comas: bool = True) -> pd.Series:
    # Misma interpretación de comas que la limpieza existente, pero sin fillna/clip.
    texto = serie.astype("string").str.strip()
    if quitar_comas:
        texto = texto.str.replace(",", "", regex=False)
    valores = pd.to_numeric(texto, errors="coerce").astype(float)
    return valores.where(np.isfinite(valores))


def _alcance(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    columnas = ["DEPARTAMENTO", "PROVINCIA", "SECTOR", "HOSPITALIZACION", "ID_HOSPITALIZACION"]
    if not set(columnas).issubset(df.columns) or df.columns.duplicated().any():
        return pd.Series(False, index=df.index), False
    textos = df[columnas].apply(limpiar_texto)
    return mascara_ipress_publicas_lima(textos) & mascara_hospitalizacion_valida(textos), True


def _resumen_base(archivo: str, filas, encoding: str, separador: str) -> dict:
    resumen = {
        "archivo": archivo, "filas_totales": filas, "filas_en_alcance_modelo": 0,
        "filas_en_alcance_modelo_periodo_valido": None,
        "encoding": encoding, "separador": separador, "estado_esquema": "OK",
        "detalle_esquema": "", "alcance_evaluable": True,
        "reglas_no_evaluadas": "", "filas_con_hallazgos": 0,
        "filas_con_hallazgos_alcance_modelo": 0,
    }
    for campo, _, _ in REGLAS.values():
        resumen[campo] = 0
        resumen[f"{campo}_alcance_modelo"] = 0
    return resumen


def _hallazgo_archivo(archivo: str, descripcion: str) -> dict:
    hallazgo = dict.fromkeys(COLUMNAS_HALLAZGOS)
    hallazgo.update(archivo=archivo, regla="Q00", severidad="ERROR", descripcion=descripcion,
                    valores_relevantes=json.dumps({"detalle": descripcion}, ensure_ascii=False))
    return hallazgo


def auditar_dataframe(
    original: pd.DataFrame, archivo: str = "datos.csv", *, encoding: str = "",
    separador: str = "",
) -> ResultadoArchivo:
    """Inspecciona una copia; índices y celdas del argumento permanecen intactos.

    Para conservar literales como NA, null y vacíos, leer con conservar_originales=True.
    Q01/Q02 cuentan celdas; Q03-Q09 cuentan filas (Q04, repeticiones posteriores).
    """
    df = normalizar_columnas(original).reset_index(drop=True)
    resumen = _resumen_base(archivo, len(df), encoding, separador)
    alcance, evaluable = _alcance(df)
    resumen.update(filas_en_alcance_modelo=int(alcance.sum()), alcance_evaluable=evaluable)
    hallazgos = []
    filas_marcadas = pd.Series(False, index=df.index)

    def registrar(regla, mascara, columnas, calculados=None):
        mascara = mascara.fillna(False)
        campo, severidad, descripcion = REGLAS[regla]
        resumen[campo] += int(mascara.sum())
        resumen[f"{campo}_alcance_modelo"] += int((mascara & alcance).sum())
        filas_marcadas.loc[mascara] = True
        seleccion = df.loc[mascara]
        for indice, fila in zip(seleccion.index, seleccion.to_dict("records")):
            valores = {c: _valor_json(fila[c]) for c in columnas}
            for c, serie in (calculados or {}).items():
                valores[c] = _valor_json(serie.loc[indice])
            hallazgos.append({
                "archivo": archivo, "fila_csv_aproximada": int(indice) + 2,
                "regla": regla, "severidad": severidad, "descripcion": descripcion,
                **{destino: _valor_json(fila.get(origen)) for destino, origen in CONTEXTOS.items()},
                "valores_relevantes": json.dumps(valores, ensure_ascii=False, allow_nan=False),
                "en_alcance_modelo": bool(alcance.loc[indice]),
            })

    def terminar():
        resumen["filas_con_hallazgos"] = int(filas_marcadas.sum())
        resumen["filas_con_hallazgos_alcance_modelo"] = int((filas_marcadas & alcance).sum())
        return ResultadoArchivo(resumen, pd.DataFrame(hallazgos, columns=COLUMNAS_HALLAZGOS))

    try:
        validar_columnas(df, archivo)
        if df.columns.duplicated().any():
            raise ValueError("Nombres de columnas duplicados después de normalizar.")
        for alias, canonica in ALIAS_COLUMNAS.items():
            if alias in df and canonica in df:
                iguales = df[alias].eq(df[canonica]) | _numeros(df[alias]).eq(_numeros(df[canonica]))
                if not iguales.fillna(False).all():
                    raise ValueError(f"Aliases contradictorios: {alias} y {canonica}; no se elige uno silenciosamente.")
    except ValueError as error:
        resumen.update(estado_esquema="ERROR", detalle_esquema=str(error), errores_esquema=1,
                       reglas_no_evaluadas="Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09")
        hallazgos.append(_hallazgo_archivo(archivo, str(error)))
        return terminar()

    numericos = {c: _numeros(df[c]) for c in COLUMNAS_NUMERICAS}
    numericos.update({c: _numeros(df[c], quitar_comas=False) for c in ("ANHO", "MES")})
    for columna, valores in numericos.items():
        registrar("Q01", valores.isna(), [columna])
        if columna in COLUMNAS_NUMERICAS:
            registrar("Q02", valores.lt(0), [columna])

    anio, mes = numericos["ANHO"], numericos["MES"]
    periodo_valido = (anio.between(ANIO_MINIMO, ANIO_MAXIMO) & anio.mod(1).eq(0)
                      & mes.between(1, 12) & mes.mod(1).eq(0))
    registrar("Q03", ~periodo_valido, ["ANHO", "MES"])
    resumen["filas_en_alcance_modelo_periodo_valido"] = int((alcance & periodo_valido).sum())
    registrar("Q04", df.duplicated(keep="first"), list(df.columns))

    camas, pacientes, disponibles = numericos[CAMAS], numericos[PACIENTES_DIA], numericos[DIAS_CAMA]
    actividad_cols = ["NRO_TOTAL_HOSPIT_ING", "NRO_TOTAL_HOSPIT_EGR", PACIENTES_DIA, "NRO_TOTAL_ESTANCIAS"]
    actividad = pd.DataFrame({c: numericos[c] for c in actividad_cols}).gt(0).any(axis=1)
    registrar("Q05", camas.eq(0) & actividad, [CAMAS, *actividad_cols])
    registrar("Q06", pacientes.gt(0) & disponibles.eq(0), [PACIENTES_DIA, DIAS_CAMA])

    dias = pd.Series(np.nan, index=df.index)
    pares = pd.DataFrame({"anio": anio, "mes": mes}).loc[periodo_valido].drop_duplicates()
    calendario = {(int(a), int(m)): monthrange(int(a), int(m))[1] for a, m in pares.itertuples(index=False, name=None)}
    dias.loc[periodo_valido] = [calendario[(int(a), int(m))] for a, m in zip(anio[periodo_valido], mes[periodo_valido])]
    patron = (periodo_valido & pacientes.gt(0) & pacientes.mod(1).eq(0) & camas.mod(1).eq(0)
              & np.isclose(camas, pacientes * dias, rtol=0, atol=1e-9))
    registrar("Q07", patron, [PACIENTES_DIA, CAMAS, "ANHO", "MES"], {"dias_mes": dias})

    ocupacion = pacientes.where(pacientes.ge(0)) / disponibles.where(disponibles.gt(0))
    registrar("Q08", ocupacion.gt(UMBRAL_OCUPACION), [PACIENTES_DIA, DIAS_CAMA], {"ocupacion_auditada": ocupacion})
    teoricos = camas.where(camas.ge(0)) * dias
    relacion = disponibles.where(disponibles.ge(0)) / teoricos.where(teoricos.gt(0))
    extremo = (relacion.lt(RELACION_DIAS_CAMA_MINIMA) | relacion.gt(RELACION_DIAS_CAMA_MAXIMA)
               | (teoricos.eq(0) & disponibles.gt(0)))
    registrar("Q09", extremo, [CAMAS, DIAS_CAMA, "ANHO", "MES"],
              {"dias_mes": dias, "dias_cama_teoricos": teoricos, "relacion_dias_cama": relacion})
    return terminar()


def _error_lectura(archivo: Path, error: Exception, encoding: str, separador: str) -> ResultadoArchivo:
    resumen = _resumen_base(archivo.name, None, encoding, separador)
    resumen.update(estado_esquema="ERROR_LECTURA", detalle_esquema=str(error),
                   errores_esquema=1, alcance_evaluable=False,
                   reglas_no_evaluadas="Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09")
    return ResultadoArchivo(resumen, pd.DataFrame([_hallazgo_archivo(archivo.name, str(error))], columns=COLUMNAS_HALLAZGOS))


def auditar_directorio(raw_dir: Path = RAW_DATA_DIR, quality_dir: Path = QUALITY_DIR) -> ResultadoAuditoria:
    """Audita todos los CSV no temporales; solo escribe en quality_dir.

    Los errores Q00 excluyen archivos completos de la preparación posterior.
    Ninguna otra regla selecciona, elimina o repara filas.
    """
    raw_dir, quality_dir = Path(raw_dir).resolve(), Path(quality_dir).resolve()
    for destino in [quality_dir, *(quality_dir / n for n in NOMBRES_REPORTES)]:
        if destino.resolve().is_relative_to(raw_dir):
            raise ValueError("Los reportes no pueden escribirse dentro de data/raw ni sobre sus archivos.")
    archivos = listar_archivos_csv(raw_dir)
    quality_dir.mkdir(parents=True, exist_ok=True)
    resumenes, validos = [], []
    with (quality_dir / "hallazgos_calidad.csv").open("w", encoding="utf-8-sig", newline="") as salida:
        pd.DataFrame(columns=COLUMNAS_HALLAZGOS).to_csv(salida, index=False)
        for archivo in archivos:
            encoding, separador = "", ""
            try:
                encoding, separador = _detectar_formato(archivo)
                df = leer_csv(archivo, conservar_originales=True)
            except (OSError, ValueError, UnicodeError) as error:
                resultado = _error_lectura(archivo, error, encoding, separador)
            else:
                resultado = auditar_dataframe(df, archivo.name, encoding=encoding, separador=separador)
            resumenes.append(resultado.resumen)
            resultado.hallazgos.to_csv(salida, index=False, header=False)
            if resultado.resumen["estado_esquema"] == "OK":
                validos.append(archivo)
            print(f"Auditoría {archivo.name}: {resultado.resumen['estado_esquema']}, {len(resultado.hallazgos)} hallazgos", flush=True)

    campos = ["filas_totales", "filas_en_alcance_modelo", "filas_en_alcance_modelo_periodo_valido",
              "filas_con_hallazgos", "filas_con_hallazgos_alcance_modelo"]
    campos += [c for campo, _, _ in REGLAS.values() for c in (campo, f"{campo}_alcance_modelo")]
    resumen = {
        "version": 1, "generado_utc": datetime.now(timezone.utc).isoformat(),
        "directorio_raw": str(raw_dir), "cantidad_archivos": len(archivos),
        "archivos_esquema_valido": len(validos), "archivos_bloqueados": len(archivos) - len(validos),
        "archivos_con_conteo_desconocido": sum(r["filas_totales"] is None for r in resumenes),
        "totales": {c: sum(r[c] or 0 for r in resumenes) for c in campos},
        "criterios": {
            "alcance_modelo": "LIMA/LIMA, sectores públicos y hospitalización del pipeline; antes de deduplicar, validar período y construir target",
            "unidades": "Q01/Q02: celdas; Q03-Q09: filas; Q04: repeticiones después de la primera dentro de cada archivo; Q00: archivos",
            "dias_cama": "DIAS_CAMA_DISPONIBLE y NRO_TOTAL_CAMAS_DISPONIB son días-cama disponibles, no camas libres",
            "periodo": {"anio_minimo": ANIO_MINIMO, "anio_maximo": ANIO_MAXIMO},
            "Q07": "Igualdad exacta camas = pacientes-día enteros positivos × días calendario; hipótesis sin corrección",
            "Q08_ocupacion_mayor_que": UMBRAL_OCUPACION,
            "Q09_relacion_fuera_de": [RELACION_DIAS_CAMA_MINIMA, RELACION_DIAS_CAMA_MAXIMA],
            "politica": "Solo Q00 bloquea archivos; las demás reglas no cambian la limpieza existente",
            "filas_csv": "Índice del registro leído + 2; aproximado si hay líneas vacías o campos multilínea",
            "no_evaluadas": "Un conteo cero con regla no evaluada no demuestra ausencia de problemas; ver cada archivo",
        },
        "reglas": {k: {"metrica": v[0], "severidad": v[1], "descripcion": v[2]} for k, v in REGLAS.items()},
        "archivos": resumenes,
    }
    pd.DataFrame(resumenes).to_csv(quality_dir / "resumen_calidad.csv", index=False, encoding="utf-8-sig")
    (quality_dir / "resumen_calidad.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Reportes de calidad guardados en {quality_dir}")
    return ResultadoAuditoria(validos, resumen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=QUALITY_DIR)
    args = parser.parse_args()
    try:
        resultado = auditar_directorio(args.raw_dir, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error de auditoría: {error}\n")
    return 1 if resultado.resumen["archivos_bloqueados"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
