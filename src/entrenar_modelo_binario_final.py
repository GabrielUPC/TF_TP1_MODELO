"""Entrena y serializa el candidato binario v1 sin utilizar targets 2026."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd

from .backtesting_binario import (
    CONFIGURACION, OBJETIVO, VERSION_TARGET, crear_pipeline_binario,
    obtener_modelos_binarios,
)
from .evaluar_holdout_2025_binario import _leer_filas, _metadata_valida, leer_sin_etiquetas
from .features_binarias import COLUMNAS_PREDICTORAS_BINARIAS, validar_predictoras
from .inferencia_binaria import (
    SIGNIFICADO_PROBABILIDAD, TARGET, THRESHOLD_OPERACIONAL, VERSION_MODELO,
    predecir_alta_presion, sha256_archivo,
)

ROOT = Path(__file__).resolve().parents[1]
MODELO_SALIDA = ROOT / "models/modelo_ipress_binario_v1.joblib"
METADATA_SALIDA = ROOT / "models/model_metadata_binario_v1.json"
LEGACY = (ROOT / "models/modelo_ipress.joblib", ROOT / "models/model_metadata.json")


def _firma(path):
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _huellas_protegidas(root, excluidos):
    excluidos = {Path(p).resolve() for p in excluidos}
    huellas = {}
    for carpeta in (Path(root)/"data", Path(root)/"models"):
        if not carpeta.exists():
            continue
        for path in carpeta.rglob("*"):
            if path.is_file() and path.resolve() not in excluidos:
                huellas[str(path.resolve())] = _firma(path)
    return huellas


def _versiones():
    paquetes = {}
    for nombre in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib"):
        try:
            paquetes[nombre] = importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError:
            paquetes[nombre] = None
    return {"python": platform.python_version(), **paquetes}


def auditar_dataset_final(dataset_path, metadata_path):
    """Lee etiquetas solo para targets <=2025; 2026 permanece sin inspeccionar."""
    _metadata_valida(dataset_path, metadata_path)
    base = leer_sin_etiquetas(dataset_path)
    actual = pd.PeriodIndex(base.periodo_actual.astype(str), freq="M")
    predicho = pd.PeriodIndex(base.periodo_predicho.astype(str), freq="M")
    if actual.hasnans or predicho.hasnans or not (predicho == actual + 1).all():
        raise ValueError("Hay saltos temporales distintos de t a t+1.")
    train = predicho.year <= 2025
    reservado = predicho.year >= 2026
    if not train.any() or str(predicho[train].max()) != "2025-12":
        raise ValueError("El entrenamiento final no termina en target 2025-12.")
    claves = ["codigo_ipress", "servicio_hospitalizacion", "periodo_actual"]
    duplicados = int(base.duplicated(claves).sum())
    if duplicados:
        raise ValueError("Hay IPRESS-servicio-mes duplicados.")
    columnas_y = [OBJETIVO, "ocupacion_siguiente_mes"]
    etiquetas = _leer_filas(dataset_path, np.flatnonzero(train), columnas_y)
    y = pd.to_numeric(etiquetas[OBJETIVO], errors="raise").to_numpy(dtype=int)
    ocupacion = pd.to_numeric(etiquetas.ocupacion_siguiente_mes, errors="raise").to_numpy(float)
    if (not np.isin(y, [0, 1]).all() or not np.isfinite(ocupacion).all()
            or not np.array_equal(y, (ocupacion > .90).astype(int))):
        raise ValueError("El target hasta 2025 no coincide con ocupación t+1 > 0.90.")
    features = validar_predictoras(COLUMNAS_PREDICTORAS_BINARIAS)
    auditoria = {
        "numero_total_registros_dataset": int(len(base)),
        "numero_observaciones_entrenamiento": int(train.sum()),
        "positivos": int(y.sum()), "negativos": int(len(y)-y.sum()),
        "prevalencia": float(y.mean()),
        "periodo_predicho_minimo": str(predicho[train].min()),
        "periodo_predicho_maximo": str(predicho[train].max()),
        "periodo_features_minimo": str(actual[train].min()),
        "periodo_features_maximo": str(actual[train].max()),
        "duplicados_ipress_servicio_mes": duplicados,
        "saltos_temporales": int((predicho != actual + 1).sum()),
        "numero_features": len(features), "lista_features": features,
        "registros_excluidos_target_2026": int(reservado.sum()),
        "targets_2026_leidos": False, "2026_utilizado": False,
    }
    return base, train, y, auditoria


def cargar_evidencia_holdout(holdout_dir, dataset_path):
    holdout_dir = Path(holdout_dir)
    congelacion = json.loads((holdout_dir/"congelacion_preholdout_2025.json").read_text(encoding="utf-8"))
    verificacion = json.loads((holdout_dir/"verificacion_holdout_2025.json").read_text(encoding="utf-8"))
    auditoria = json.loads((holdout_dir/"auditoria_temporal_holdout_2025.json").read_text(encoding="utf-8"))
    if (congelacion.get("dataset_sha256") != sha256_archivo(dataset_path)
            or congelacion.get("modelo_principal") != "XGBoost"
            or congelacion.get("threshold_principal") != THRESHOLD_OPERACIONAL
            or congelacion.get("features") != COLUMNAS_PREDICTORAS_BINARIAS
            or congelacion.get("configuraciones", {}).get("XGBoost") != CONFIGURACION["XGBoost"]
            or not verificacion.get("auditoria_preapertura_aprobada")
            or verificacion.get("aperturas_etiqueta_2025") != 1
            or verificacion.get("2026_evaluado") or verificacion.get("2026_predicho")
            or not auditoria.get("metricas_2025_calculadas")):
        raise ValueError("La evidencia del holdout sellado no acredita el contrato congelado.")
    for nombre, esperado in verificacion.get("archivos", {}).items():
        if sha256_archivo(holdout_dir/nombre) != esperado:
            raise ValueError(f"Cambió la evidencia sellada: {nombre}.")
    raw = pd.read_csv(holdout_dir/"metricas_holdout_2025.csv").iloc[0].to_dict()
    decision = pd.read_csv(holdout_dir/"metricas_threshold_036_2025.csv").iloc[0].to_dict()
    return {
        "desarrollo_2018_2024": {
            "descripcion": "backtesting temporal expansivo previo a 2025",
            "probabilidades_sha256": congelacion["probabilidades_desarrollo_sha256"],
            "seleccion_realizada_con": congelacion["seleccion_realizada_con"],
        },
        "holdout_sellado_2025": {
            "directorio": str(holdout_dir.resolve()),
            "verificacion_sha256": sha256_archivo(holdout_dir/"verificacion_holdout_2025.json"),
            "n": int(raw["n"]), "positivos": int(raw["positivos"]),
            "negativos": int(raw["negativos"]), "prevalencia": float(raw["prevalencia_real"]),
            "roc_auc": float(raw["roc_auc"]), "pr_auc": float(raw["average_precision"]),
            "brier": float(raw["brier_score"]), "ece_10_bins": float(raw["ece_10_bins"]),
            "threshold": float(decision["threshold"]), "recall": float(decision["recall"]),
            "especificidad": float(decision["especificidad"]),
            "precision": float(decision["precision_ppv"]), "f1": float(decision["f1"]),
            "fn": int(decision["fn"]), "fp": int(decision["fp"]),
            "tasa_alertas": float(decision["tasa_alertas"]),
        },
    }


def entrenar_y_serializar(dataset_path, metadata_dataset_path, holdout_dir,
                          modelo_path=MODELO_SALIDA, metadata_path=METADATA_SALIDA,
                          *, algoritmo=None, constructor_pipeline=None, evidencia=None,
                          root=ROOT):
    """Entrena una vez, valida fuera del repo y publica dos archivos nuevos."""
    modelo_path, metadata_path, root = Path(modelo_path), Path(metadata_path), Path(root)
    legacy_root = {p.resolve() for p in (
        root/"models/modelo_ipress.joblib", root/"models/model_metadata.json")}
    if modelo_path.resolve() in legacy_root or metadata_path.resolve() in legacy_root:
        raise ValueError("Está prohibido sobrescribir los artefactos legacy.")
    if modelo_path.exists() or metadata_path.exists():
        raise FileExistsError("El candidato binario v1 ya existe; no se sobrescribe.")
    if modelo_path.parent.resolve() != metadata_path.parent.resolve():
        raise ValueError("Modelo y metadata deben compartir directorio.")
    print("1/4 Auditoría del dataset final", flush=True)
    base, train, y, auditoria = auditar_dataset_final(dataset_path, metadata_dataset_path)
    evidencia = cargar_evidencia_holdout(holdout_dir, dataset_path) if evidencia is None else evidencia
    X = base.loc[train, COLUMNAS_PREDICTORAS_BINARIAS].copy()
    for columna in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[columna] = X[columna].astype(object).where(X[columna].notna(), np.nan)
    algoritmo = obtener_modelos_binarios()["XGBoost"] if algoritmo is None else algoritmo
    constructor = constructor_pipeline or crear_pipeline_binario
    print("2/4 Firma estructural de archivos protegidos", flush=True)
    protegidos_antes = _huellas_protegidas(root, (modelo_path, metadata_path))
    pipeline = constructor(X, algoritmo)
    print("3/4 Ajuste XGBoost congelado", flush=True)
    pipeline.fit(X, y)
    artefacto = {
        "pipeline": pipeline, "version_modelo": VERSION_MODELO, "target": TARGET,
        "threshold_operacional": THRESHOLD_OPERACIONAL,
        "features": list(COLUMNAS_PREDICTORAS_BINARIAS),
    }
    muestra = X.iloc[:min(32, len(X))].copy()
    pred_antes = predecir_alta_presion(artefacto, muestra)
    modelo_path.parent.mkdir(parents=True, exist_ok=True)
    print("4/4 Serialización y smoke test", flush=True)
    with tempfile.TemporaryDirectory(prefix="binario_v1_") as temporal:
        temporal = Path(temporal)
        modelo_tmp, metadata_tmp = temporal/modelo_path.name, temporal/metadata_path.name
        print("4.1/4 Guardando joblib temporal", flush=True)
        joblib.dump(artefacto, modelo_tmp)
        print("4.2/4 Recargando joblib temporal", flush=True)
        recargado = joblib.load(modelo_tmp)
        print("4.3/4 Comparando predicciones", flush=True)
        pred_despues = predecir_alta_presion(recargado, muestra)
        probabilidades_iguales = np.allclose(
            pred_antes.probabilidad_alta_presion,
            pred_despues.probabilidad_alta_presion,
            rtol=0, atol=1e-12,
        )
        alertas_iguales = np.array_equal(
            pred_antes.alerta_alta_presion, pred_despues.alerta_alta_presion)
        if not probabilidades_iguales or not alertas_iguales:
            raise RuntimeError("La serialización modificó las predicciones.")
        print("4.4/4 Calculando SHA-256 y metadata", flush=True)
        modelo_sha = sha256_archivo(modelo_tmp)
        metadata = {
            "version_modelo": VERSION_MODELO, "tipo_modelo": "clasificacion_binaria",
            "algoritmo": "XGBClassifier", "target": TARGET,
            "definicion_target": {
                "regla": "Y_(t+1) = 1 si ocupacion_(t+1) > 0.90; 0 si <= 0.90",
                "version": VERSION_TARGET, "horizonte": "mes calendario siguiente exacto",
                "significado_probabilidad": SIGNIFICADO_PROBABILIDAD,
            },
            "threshold_operacional": THRESHOLD_OPERACIONAL,
            "numero_features": len(COLUMNAS_PREDICTORAS_BINARIAS),
            "lista_features": list(COLUMNAS_PREDICTORAS_BINARIAS),
            "periodo_entrenamiento": {
                "features_minimo": auditoria["periodo_features_minimo"],
                "features_maximo": auditoria["periodo_features_maximo"],
                "target_minimo": auditoria["periodo_predicho_minimo"],
                "target_maximo": auditoria["periodo_predicho_maximo"],
            },
            "numero_observaciones": auditoria["numero_observaciones_entrenamiento"],
            "positivos": auditoria["positivos"], "negativos": auditoria["negativos"],
            "prevalencia": auditoria["prevalencia"],
            "hiperparametros_xgboost": dict(CONFIGURACION["XGBoost"]), "seed": 42,
            "versiones": _versiones(),
            "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
            "modelo_sha256": modelo_sha,
            "dataset_sha256": sha256_archivo(dataset_path),
            "dataset_metadata_sha256": sha256_archivo(metadata_dataset_path),
            "auditoria_dataset_final": auditoria,
            "referencias_validacion": evidencia,
            "advertencia_2025": (
                "2025 fue holdout sellado para seleccionar esta versión y después fue "
                "incorporado al entrenamiento final candidato."
            ),
            "2026_utilizado": False,
            "registros_2026_excluidos": auditoria["registros_excluidos_target_2026"],
            "calibracion_nueva_aplicada": False,
            "smoke_test_serializacion": {
                "n": len(muestra), "tolerancia_absoluta": 1e-12,
                "probabilidades_coinciden": bool(probabilidades_iguales),
                "alertas_coinciden": bool(alertas_iguales),
                "feature_order_verificado": list(muestra.columns) == COLUMNAS_PREDICTORAS_BINARIAS,
            },
            "contrato_inferencia": {
                "salidas": ["probabilidad_alta_presion", "threshold", "alerta_alta_presion"],
                "regla_alerta": "probabilidad_alta_presion >= 0.36",
                "columnas_adicionales_por_defecto": "rechazadas",
                "niveles_bajo_medio_alto": False,
            },
            "artefacto_candidato_no_integrado": True,
            "modelo_legacy_sobrescrito": False,
        }
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print("4.5/4 Publicando candidato verificado", flush=True)
        try:
            shutil.copyfile(modelo_tmp, modelo_path)
            if sha256_archivo(modelo_path) != modelo_sha:
                raise RuntimeError("El SHA-256 cambió al publicar el candidato.")
            shutil.copyfile(metadata_tmp, metadata_path)
        except Exception:
            modelo_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
    print("4.6/4 Verificando archivos protegidos", flush=True)
    protegidos_despues = _huellas_protegidas(root, (modelo_path, metadata_path))
    if protegidos_antes != protegidos_despues:
        modelo_path.unlink(missing_ok=True); metadata_path.unlink(missing_ok=True)
        raise RuntimeError("Cambió un archivo protegido durante el entrenamiento.")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--modelo-salida", type=Path, default=MODELO_SALIDA)
    parser.add_argument("--metadata-salida", type=Path, default=METADATA_SALIDA)
    args = parser.parse_args(argv)
    metadata = entrenar_y_serializar(
        args.dataset, args.dataset_metadata, args.holdout_dir,
        args.modelo_salida, args.metadata_salida,
    )
    print(json.dumps({k: metadata[k] for k in (
        "version_modelo", "numero_observaciones", "positivos", "negativos",
        "prevalencia", "modelo_sha256", "2026_utilizado")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
