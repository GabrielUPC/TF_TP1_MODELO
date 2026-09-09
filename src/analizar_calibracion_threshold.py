"""Calibracion cruda y thresholds binarios sobre predicciones OOF 2018-2024.

No ajusta modelos ni calibradores. Los escenarios walk-forward seleccionan su
threshold exclusivamente con predicciones out-of-fold de años anteriores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtesting_binario import calcular_metricas_binarias, validar_salida_externa

MODELOS = ["Regresion_Logistica", "Random_Forest", "XGBoost"]
ANIOS_DESARROLLO = list(range(2018, 2025))
GRID_THRESHOLDS = np.round(np.arange(0.05, 0.801, 0.01), 2)
ESCENARIOS = [
    "max_f1", "max_balanced_accuracy", "recall_080", "recall_085",
    "recall_090", "referencia_050",
]
ARCHIVOS = [
    "calibracion_global.csv", "calibracion_por_anio.csv",
    "calibracion_por_familia.csv", "metricas_por_threshold.csv",
    "metricas_threshold_por_anio.csv", "metricas_threshold_por_familia.csv",
    "escenarios_operacionales.csv", "escenarios_validacion_temporal.csv",
    "resumen_validacion_temporal_threshold.csv",
    "comparacion_final_candidatos.csv", "metodologia_calibracion_threshold.json",
    "resumen_metodologico.md",
    "escenarios_operacionales_por_anio.csv",
    "escenarios_operacionales_por_familia.csv",
]


def sha256_archivo(path: Path) -> str:
    with Path(path).open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def validar_probabilidades(df: pd.DataFrame) -> pd.DataFrame:
    requeridas = {
        "indice_dataset", "modelo", "tipo_score", "anio_prueba",
        "familia_servicio", "y_real", "score_positivo", "codigo_ipress",
        "servicio_hospitalizacion", "periodo_actual", "periodo_predicho",
    }
    faltantes = sorted(requeridas.difference(df.columns))
    if faltantes:
        raise ValueError("Faltan columnas OOF: " + ", ".join(faltantes))
    resultado = df.copy()
    esenciales = ["indice_dataset", "modelo", "anio_prueba", "familia_servicio",
                  "y_real", "score_positivo"]
    if resultado.empty or resultado[esenciales].isna().any().any():
        raise ValueError("Las predicciones OOF están vacías o contienen valores ausentes.")
    resultado["anio_prueba"] = pd.to_numeric(resultado.anio_prueba, errors="raise").astype(int)
    if resultado.anio_prueba.ge(2025).any():
        raise ValueError("2025/2026 están reservados y no pueden abrirse en este análisis.")
    if set(resultado.anio_prueba.unique()) != set(ANIOS_DESARROLLO):
        raise ValueError("Se requieren exactamente los años OOF 2018-2024.")
    if set(resultado.modelo.unique()) != set(MODELOS):
        raise ValueError("Se requieren exactamente Logistic, Random Forest y XGBoost.")
    if set(resultado.tipo_score.unique()) != {"probabilidad_sin_calibracion"}:
        raise ValueError("Solo se admiten probabilidades OOF crudas de los modelos.")
    resultado["y_real"] = pd.to_numeric(resultado.y_real, errors="raise").astype(int)
    resultado["score_positivo"] = pd.to_numeric(resultado.score_positivo, errors="raise")
    if (not resultado.y_real.isin([0, 1]).all()
            or not np.isfinite(resultado.score_positivo).all()
            or not resultado.score_positivo.between(0, 1).all()):
        raise ValueError("Target o probabilidades OOF inválidos.")
    if not resultado.familia_servicio.isin(["hospitalizacion", "cuidados_criticos"]).all():
        raise ValueError("Familia distinta de 24xx/25xx.")
    claves = ["modelo", "anio_prueba", "indice_dataset"]
    if resultado.duplicated(claves).any():
        raise ValueError("Predicciones OOF duplicadas por modelo, año e índice.")
    referencia = None
    columnas_registro = ["anio_prueba", "indice_dataset", "y_real", "familia_servicio",
                         "codigo_ipress", "servicio_hospitalizacion",
                         "periodo_actual", "periodo_predicho"]
    for modelo in MODELOS:
        actual = resultado.loc[resultado.modelo.eq(modelo), columnas_registro].sort_values(
            ["anio_prueba", "indice_dataset"], kind="stable"
        ).reset_index(drop=True)
        if referencia is None:
            referencia = actual
        elif not actual.equals(referencia):
            raise ValueError("Los modelos no usan exactamente los mismos registros OOF.")
    return resultado


def metricas_threshold(y_real, score_positivo, threshold: float) -> dict:
    y = np.asarray(y_real)
    score = np.asarray(score_positivo, dtype=float)
    if (y.ndim != 1 or score.shape != y.shape or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(score).all()
            or (score < 0).any() or (score > 1).any()
            or not 0 < float(threshold) < 1):
        raise ValueError("Datos o threshold inválidos.")
    y = y.astype(int)
    pred = score >= float(threshold)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp, fn = neg - tn, pos - tp
    recall = tp / pos if pos else np.nan
    especificidad = tn / neg if neg else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    return {
        "threshold": float(threshold), "n": len(y), "positivos": pos,
        "negativos": neg, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "recall": recall, "especificidad": especificidad,
        "precision_ppv": precision, "npv": npv, "f1": f1,
        "balanced_accuracy": float(np.nanmean([recall, especificidad])),
        "accuracy": (tp + tn) / len(y), "tasa_alertas": float(pred.mean()),
        "tasa_falsos_negativos": fn / pos if pos else np.nan,
        "prevalencia_real": pos / len(y),
        "brier_score": float(np.mean((score - y) ** 2)),
    }


def tabla_thresholds(df: pd.DataFrame, dimensiones=()) -> pd.DataFrame:
    grupos = [((), df)] if not dimensiones else df.groupby(list(dimensiones), sort=True)
    filas = []
    for clave, grupo in grupos:
        clave = clave if isinstance(clave, tuple) else (clave,)
        prefijo = dict(zip(dimensiones, clave))
        for threshold in GRID_THRESHOLDS:
            filas.append({**prefijo, **metricas_threshold(
                grupo.y_real.to_numpy(), grupo.score_positivo.to_numpy(), threshold
            )})
    return pd.DataFrame(filas)


def tabla_calibracion(df: pd.DataFrame, dimensiones=()) -> pd.DataFrame:
    grupos = [((), df)] if not dimensiones else df.groupby(list(dimensiones), sort=True)
    filas = []
    for clave, grupo in grupos:
        clave = clave if isinstance(clave, tuple) else (clave,)
        prefijo = dict(zip(dimensiones, clave))
        y = grupo.y_real.to_numpy(dtype=int)
        score = grupo.score_positivo.to_numpy(dtype=float)
        bins = np.minimum((score * 10).astype(int), 9)
        brier = float(np.mean((score - y) ** 2))
        for numero in range(10):
            mascara = bins == numero
            media = float(score[mascara].mean()) if mascara.any() else np.nan
            observada = float(y[mascara].mean()) if mascara.any() else np.nan
            filas.append({
                **prefijo, "bin": numero, "limite_inferior": numero / 10,
                "limite_superior": (numero + 1) / 10, "n": int(mascara.sum()),
                "probabilidad_media": media, "prevalencia_observada": observada,
                "sesgo_bin": media - observada if mascara.any() else np.nan,
                "brier_grupo": brier,
            })
    return pd.DataFrame(filas)


def resumen_calibracion(tabla: pd.DataFrame, dimensiones=("modelo",)) -> pd.DataFrame:
    filas = []
    for clave, grupo in tabla.groupby(list(dimensiones), sort=True):
        clave = clave if isinstance(clave, tuple) else (clave,)
        n = grupo.n.sum()
        filas.append({
            **dict(zip(dimensiones, clave)), "n": int(n),
            "brier": grupo.brier_grupo.iloc[0],
            "ece_10_bins": float((grupo.n * grupo.sesgo_bin.abs()).sum() / n),
            "sesgo_calibracion": float((grupo.n * grupo.sesgo_bin).sum() / n),
        })
    return pd.DataFrame(filas)


def seleccionar_escenarios(tabla: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for modelo, grupo in tabla.groupby("modelo", sort=True):
        def guardar(escenario, fila, objetivo, viable=True):
            valores = ({k: fila[k] for k in fila.index} if viable else {
                k: np.nan for k in grupo.columns if k != "modelo"})
            filas.append({"modelo": modelo, "escenario": escenario,
                          "objetivo_escenario": objetivo, "viable": viable, **valores})

        mejor_f1 = grupo.sort_values(
            ["f1", "balanced_accuracy", "tasa_alertas", "threshold"],
            ascending=[False, False, True, False], kind="stable").iloc[0]
        guardar("max_f1", mejor_f1, "Máximo F1; desempate por balanced accuracy y menor alerta")
        mejor_bal = grupo.sort_values(
            ["balanced_accuracy", "f1", "tasa_alertas", "threshold"],
            ascending=[False, False, True, False], kind="stable").iloc[0]
        guardar("max_balanced_accuracy", mejor_bal,
                "Máxima balanced accuracy; desempate por F1 y menor alerta")
        for minimo, nombre in [(0.80, "recall_080"), (0.85, "recall_085"), (0.90, "recall_090")]:
            factibles = grupo.loc[grupo.recall.ge(minimo)].sort_values(
                ["threshold", "precision_ppv"], ascending=[False, False], kind="stable")
            guardar(nombre, factibles.iloc[0] if len(factibles) else grupo.iloc[0],
                    f"Mayor threshold de la grilla con recall >= {minimo:.2f}", bool(len(factibles)))
        referencia = grupo.loc[np.isclose(grupo.threshold, 0.50)].iloc[0]
        guardar("referencia_050", referencia, "Umbral técnico provisional 0.50")
    return pd.DataFrame(filas)


def validacion_temporal_escenarios(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    anios = sorted(df.anio_prueba.unique())
    for anio in anios[1:]:
        historico = df.loc[df.anio_prueba.lt(anio)]
        prueba = df.loc[df.anio_prueba.eq(anio)]
        escenarios = seleccionar_escenarios(tabla_thresholds(historico, ("modelo",)))
        for _, escenario in escenarios.iterrows():
            if not bool(escenario.viable):
                continue
            actual = prueba.loc[prueba.modelo.eq(escenario.modelo)]
            metrica = metricas_threshold(actual.y_real, actual.score_positivo,
                                         escenario.threshold)
            filas.append({
                "modelo": escenario.modelo, "escenario": escenario.escenario,
                "anio_evaluacion": int(anio),
                "anios_seleccion": ",".join(map(str, sorted(historico.anio_prueba.unique()))),
                "anio_maximo_seleccion": int(historico.anio_prueba.max()),
                "n_seleccion": int(len(historico.loc[historico.modelo.eq(escenario.modelo)])),
                **metrica,
            })
    return pd.DataFrame(filas)


def evaluar_escenarios_fijos(escenarios: pd.DataFrame,
                             tabla_detallada: pd.DataFrame) -> pd.DataFrame:
    """Extrae cada threshold candidato global sin volver a seleccionarlo."""
    seleccion = escenarios.loc[escenarios.viable,
                                ["modelo", "escenario", "threshold"]]
    return seleccion.merge(tabla_detallada, on=["modelo", "threshold"],
                            how="inner", validate="many_to_many")


def resumir_validacion_temporal(tabla: pd.DataFrame) -> pd.DataFrame:
    metricas = ["recall", "especificidad", "precision_ppv", "npv", "f1",
                "balanced_accuracy", "tasa_alertas", "tasa_falsos_negativos"]
    filas = []
    for (modelo, escenario), grupo in tabla.groupby(["modelo", "escenario"], sort=True):
        fila = {"modelo": modelo, "escenario": escenario, "n_anios": len(grupo)}
        for metrica in metricas:
            fila[f"{metrica}_promedio"] = grupo[metrica].mean()
            fila[f"{metrica}_min"] = grupo[metrica].min()
            fila[f"{metrica}_max"] = grupo[metrica].max()
        fila["fp_total"] = int(grupo.fp.sum())
        fila["fn_total"] = int(grupo.fn.sum())
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_candidatos(df: pd.DataFrame, calibracion: pd.DataFrame,
                        escenarios: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    cal = resumen_calibracion(calibracion).set_index("modelo")
    filas = []
    for modelo in MODELOS:
        actual = df.loc[df.modelo.eq(modelo)]
        discriminacion = calcular_metricas_binarias(actual.y_real, actual.score_positivo)
        anuales = pd.DataFrame([
            calcular_metricas_binarias(g.y_real, g.score_positivo)
            for _, g in actual.groupby("anio_prueba", sort=True)
        ])
        familias = pd.DataFrame([
            calcular_metricas_binarias(g.y_real, g.score_positivo)
            for _, g in actual.groupby("familia_servicio", sort=True)
        ])
        esc = escenarios.loc[escenarios.modelo.eq(modelo)].set_index("escenario")
        val = temporal.loc[(temporal.modelo.eq(modelo)) & temporal.escenario.eq("max_f1")]
        filas.append({
            "modelo": modelo, "n_oof": len(actual),
            "roc_auc_oof": discriminacion["roc_auc"],
            "average_precision_oof": discriminacion["average_precision"],
            "brier_oof": cal.loc[modelo, "brier"],
            "ece_10_bins_oof": cal.loc[modelo, "ece_10_bins"],
            "sesgo_calibracion_oof": cal.loc[modelo, "sesgo_calibracion"],
            "average_precision_anual_std": anuales.average_precision.std(ddof=0),
            "average_precision_anual_min": anuales.average_precision.min(),
            "brier_anual_std": anuales.brier_score.std(ddof=0),
            "brier_anual_max": anuales.brier_score.max(),
            "average_precision_familia_min": familias.average_precision.min(),
            "brier_familia_max": familias.brier_score.max(),
            "threshold_max_f1_desarrollo": esc.loc["max_f1", "threshold"],
            "f1_max_desarrollo": esc.loc["max_f1", "f1"],
            "recall_max_f1_desarrollo": esc.loc["max_f1", "recall"],
            "tasa_alertas_max_f1_desarrollo": esc.loc["max_f1", "tasa_alertas"],
            "f1_walk_forward_promedio": val.f1.mean(),
            "f1_walk_forward_std": val.f1.std(ddof=0),
            "f1_walk_forward_min": val.f1.min(),
            "recall_walk_forward_promedio": val.recall.mean(),
            "recall_walk_forward_std": val.recall.std(ddof=0),
            "recall_walk_forward_min": val.recall.min(),
            "tasa_alertas_walk_forward_promedio": val.tasa_alertas.mean(),
            "tasa_alertas_walk_forward_max": val.tasa_alertas.max(),
            "threshold_recall_085_desarrollo": esc.loc["recall_085", "threshold"],
            "fn_recall_085_desarrollo": esc.loc["recall_085", "fn"],
            "fp_recall_085_desarrollo": esc.loc["recall_085", "fp"],
            "tasa_alertas_recall_085_desarrollo": esc.loc["recall_085", "tasa_alertas"],
            "2025_evaluado": False, "calibrador_ajustado": False,
            "modelo_produccion": False,
        })
    return pd.DataFrame(filas).sort_values(
        ["average_precision_oof", "roc_auc_oof"], ascending=False, kind="stable"
    ).reset_index(drop=True)


def ejecutar(probabilidades_path: Path, output_dir: Path):
    probabilidades_path = Path(probabilidades_path).resolve()
    destino = validar_salida_externa(output_dir)
    df = pd.read_csv(probabilidades_path, dtype={"codigo_ipress": "string"},
                     float_precision="round_trip")
    # El artefacto OOF original también contiene baselines deterministas. Antes
    # de filtrarlos se sella cualquier periodo reservado presente en la fuente.
    if "anio_prueba" not in df or "tipo_score" not in df:
        raise ValueError("La fuente no identifica año ni tipo de score.")
    anios_fuente = pd.to_numeric(df.anio_prueba, errors="raise")
    if anios_fuente.ge(2025).any():
        raise ValueError("2025/2026 están reservados y no pueden abrirse en este análisis.")
    df = df.loc[df.tipo_score.eq("probabilidad_sin_calibracion")].copy()
    df = validar_probabilidades(df)
    destino.mkdir(parents=True, exist_ok=True)
    cal_global = tabla_calibracion(df, ("modelo",))
    cal_anio = tabla_calibracion(df, ("modelo", "anio_prueba"))
    cal_familia = tabla_calibracion(df, ("modelo", "familia_servicio"))
    thresholds = tabla_thresholds(df, ("modelo",))
    thresholds_anio = tabla_thresholds(df, ("modelo", "anio_prueba"))
    thresholds_familia = tabla_thresholds(df, ("modelo", "familia_servicio"))
    escenarios = seleccionar_escenarios(thresholds)
    escenarios_anio = evaluar_escenarios_fijos(escenarios, thresholds_anio)
    escenarios_familia = evaluar_escenarios_fijos(escenarios, thresholds_familia)
    temporal = validacion_temporal_escenarios(df)
    resumen_temporal = resumir_validacion_temporal(temporal)
    comparacion = comparar_candidatos(df, cal_global, escenarios, temporal)
    tablas = {
        ARCHIVOS[0]: cal_global, ARCHIVOS[1]: cal_anio, ARCHIVOS[2]: cal_familia,
        ARCHIVOS[3]: thresholds, ARCHIVOS[4]: thresholds_anio,
        ARCHIVOS[5]: thresholds_familia, ARCHIVOS[6]: escenarios,
        ARCHIVOS[7]: temporal, ARCHIVOS[8]: resumen_temporal, ARCHIVOS[9]: comparacion,
        ARCHIVOS[12]: escenarios_anio, ARCHIVOS[13]: escenarios_familia,
    }
    for nombre, tabla in tablas.items():
        tabla.to_csv(destino / nombre, index=False)
    metodologia = {
        "version": "calibracion_threshold_binario_oof_v1",
        "fuente_probabilidades": str(probabilidades_path),
        "fuente_sha256": sha256_archivo(probabilidades_path),
        "modelos": MODELOS, "anios_desarrollo": ANIOS_DESARROLLO,
        "2025_utilizado": False, "2026_utilizado": False,
        "calibracion": "Diagnóstico crudo; no se ajustó Platt, sigmoid ni isotonic",
        "calibrador_ajustado": False,
        "grid_thresholds": {"desde": 0.05, "hasta": 0.80, "paso": 0.01},
        "seleccion_walk_forward": (
            "Cada año 2019-2024 usa un threshold elegido solo con predicciones OOF de años anteriores"
        ),
        "seleccion_candidatos_futuros": (
            "Los escenarios 2018-2024 son candidatos para evaluación futura aún sellada; no son definitivos"
        ),
        "umbral_operacional_definitivo": None, "modelo_final_produccion": None,
    }
    (destino / ARCHIVOS[10]).write_text(
        json.dumps(metodologia, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destino / ARCHIVOS[11]).write_text(
        "# Calibración y thresholds binarios\n\n"
        "Análisis de probabilidades out-of-fold de 2018-2024. No se ajustaron "
        "calibradores ni modelos. 2025 y 2026 permanecieron sellados.\n\n"
        "La grilla va de 0.05 a 0.80 cada 0.01. Los escenarios walk-forward "
        "seleccionan el threshold de cada año usando exclusivamente años anteriores. "
        "Los escenarios agregados de todo desarrollo sirven como candidatos para un "
        "periodo futuro aún no evaluado, no como thresholds definitivos.\n",
        encoding="utf-8",
    )
    return tablas, metodologia


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilidades", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    ejecutar(args.probabilidades, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
