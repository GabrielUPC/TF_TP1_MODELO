"""Aparta meses Q05/Q06/Q07/Q08 usando la evidencia de auditoría RAW."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__:
    from .datos_raw import limpiar_texto, listar_archivos_csv
else:
    from datos_raw import limpiar_texto, listar_archivos_csv

VERSION_POLITICA = "capacidad_q05_q06_q07_q08_v3"
CLAVE = ["codigo_ipress", "servicio_hospitalizacion", "anio", "mes"]
RENOMBRES = dict(zip(
    ["CO_IPRESS", "HOSPITALIZACION", "ANHO", "MES"], CLAVE,
))


def sha256_archivo(path: Path) -> str:
    with Path(path).open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def huellas_raw(raw_dir: Path) -> dict[str, str]:
    return {p.name: sha256_archivo(p) for p in listar_archivos_csv(raw_dir)}


def _claves(df: pd.DataFrame) -> pd.DataFrame:
    claves = df.rename(columns=RENOMBRES)[CLAVE].copy()
    for c in CLAVE[:2]:
        claves[c] = limpiar_texto(claves[c])
    for c in CLAVE[2:]:
        numeros = pd.to_numeric(claves[c], errors="raise")
        if numeros.isna().any() or numeros.mod(1).ne(0).any():
            raise ValueError("No se pueden relacionar alertas de capacidad con periodos no enteros.")
        claves[c] = numeros.astype(int)
    if not claves.mes.between(1, 12).all():
        raise ValueError("Mes inválido en alertas de capacidad; revisar el periodo antes de preparar.")
    return claves


def apartar_meses_pendientes(df: pd.DataFrame, quality_dir: Path, *, raw_sha256: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Recibe filas limpias en alcance, pero decide usando alertas sobre RAW.

    Retira el mes completo por IPRESS y servicio si alguna fila fuente tiene
    Q05/Q06/Q07/Q08. No modifica df ni RAW. Debe llamarse antes de consolidar y derivar.
    """
    quality_dir = Path(quality_dir)
    reporte = quality_dir / "hallazgos_calidad.csv"
    if not reporte.is_file():
        raise ValueError("Falta la auditoría RAW; no se puede aplicar el tratamiento de capacidad.")
    partes = []
    for chunk in pd.read_csv(reporte, dtype=str, keep_default_na=False, chunksize=100000):
        mascara = chunk.regla.isin(["Q05", "Q06", "Q07", "Q08"]) & chunk.en_alcance_modelo.str.lower().eq("true")
        partes.append(chunk.loc[mascara])
    pendientes = pd.concat(partes, ignore_index=True) if partes else pd.read_csv(reporte, dtype=str, nrows=0)
    pendientes["estado_revision"] = "PENDIENTE_VALIDACION_NO_USAR_ENTRENAMIENTO"
    pendientes.to_csv(quality_dir / "pendientes_capacidad.csv", index=False, encoding="utf-8-sig")
    if pendientes.empty:
        claves_pendientes = pd.DataFrame(columns=CLAVE)
        retirar = pd.Series(False, index=df.index)
    else:
        claves_pendientes = _claves(pendientes).drop_duplicates()
        indice_pendientes = pd.MultiIndex.from_frame(claves_pendientes)
        retirar = pd.Series(pd.MultiIndex.from_frame(_claves(df)).isin(indice_pendientes), index=df.index)
    claves_pendientes.to_csv(quality_dir / "meses_pendientes_capacidad.csv", index=False, encoding="utf-8-sig")
    tratamiento = {
        "version": VERSION_POLITICA,
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": raw_sha256 or {},
        "auditoria_sha256": sha256_archivo(reporte),
        "reglas_aplicadas": ["Q05", "Q06", "Q07", "Q08"],
        "criterio": "Apartar mes completo por IPRESS y servicio si alguna fila RAW en alcance tiene Q05/Q06/Q07/Q08; no imputar valores.",
        "filas_raw_pendientes": int(pendientes[["archivo", "fila_csv_aproximada"]].drop_duplicates().shape[0]),
        "meses_servicio_pendientes": int(len(claves_pendientes)),
        "filas_apartadas_antes_consolidacion": int(retirar.sum()),
        "filas_restantes_antes_consolidacion": int((~retirar).sum()),
        "reglas_solo_auditadas": ["Q01", "Q02", "Q03", "Q04", "Q09"],
    }
    (quality_dir / "tratamiento_capacidad.json").write_text(json.dumps(tratamiento, ensure_ascii=False, indent=2), encoding="utf-8")
    resultado = df.loc[~retirar].copy()
    if resultado.empty:
        raise ValueError("No quedan datos después de apartar capacidad pendiente; revisar los reportes.")
    return resultado, tratamiento
