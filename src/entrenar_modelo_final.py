"""Entrenamiento final único XGBoost D, sin tuning ni reevaluar 2025.

python -m src.entrenar_modelo_final --solo-plan
python -m src.entrenar_modelo_final
python -m src.entrenar_modelo_final --actualizar-solo-metadata
No invoca entrenar_modelos(), que genera y reemplaza informes anteriores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np
import pandas as pd

from src import backtesting_temporal as bt
from src.modelo_final import (
    ANIOS_DESARROLLO, REGLA_FINAL, ModeloFinalD,
    SIGNIFICADO_PROBABILIDAD, SIGNIFICADO_INDICE,
)
from src.tratamiento_capacidad import VERSION_POLITICA, sha256_archivo
from src.variables_temporales_experimentales import CONJUNTOS, agregar_features_candidatas

ROOT = Path(__file__).resolve().parents[1]
PRODUCTIVOS = {"models/modelo_ipress.joblib", "models/model_metadata.json"}


def huellas_protegidas(root):
    """RAW, procesados y toda evidencia experimental son de solo lectura."""
    return {p.relative_to(root).as_posix(): sha256_archivo(p)
            for carpeta in ("data", "models") for p in (root / carpeta).rglob("*")
            if p.is_file() and p.relative_to(root).as_posix() not in PRODUCTIVOS}


def verificar_huellas(root, huellas):
    if any(not (root / p).is_file() or sha256_archivo(root / p) != h for p, h in huellas.items()):
        raise RuntimeError("Cambió un archivo protegido; no se debe publicar el modelo final.")


def normalizar(valor):
    if isinstance(valor, dict):
        return {k: normalizar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [normalizar(v) for v in valor]
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor


def leer_plan(root=ROOT):
    root = Path(root)
    csv = root / "data/processed/dataset_modelo_ipress.csv"
    dataset_meta = json.loads((csv.parent / "dataset_metadata.json").read_text(encoding="utf-8"))
    seleccion = json.loads((root / "models/seleccion_regla_extension_020.json").read_text(encoding="utf-8"))
    sha = sha256_archivo(csv)
    if (sha != dataset_meta.get("dataset_sha256")
            or sha != seleccion.get("procedencia", {}).get("dataset_sha256")):
        raise ValueError("Dataset D no coincide con las huellas de la evidencia congelada.")
    if dataset_meta.get("tratamiento_capacidad", {}).get("version") != VERSION_POLITICA:
        raise ValueError("El dataset no tiene la política de calidad vigente.")
    if (seleccion.get("estado") != "completado_sin_produccion"
            or seleccion.get("regla_seleccionada") != REGLA_FINAL["nombre"]
            or seleccion.get("anios_desarrollo") != ANIOS_DESARROLLO):
        raise ValueError("La evidencia no acredita la selección final solicitada.")
    columnas = seleccion["columnas_predictoras"]
    if (len(set(columnas)) != len(columnas) or set(columnas) & bt.ETIQUETAS
            or columnas[-len(CONJUNTOS['D']):] != CONJUNTOS['D']):
        raise ValueError("Contrato de features D inválido o con objetivo entre predictores.")
    df = pd.read_csv(csv, dtype={c: "string" for c in ("codigo_ipress", "ubigeo", "id_hospitalizacion")})
    bt.validar_periodos(df)
    y = df[bt.OBJETIVO]
    if set(y.unique()) != {0, 1, 2}:
        raise ValueError("El entrenamiento final debe contener las tres clases.")
    # Conserva TODOS los registros elegibles, incluso con lags D ausentes.
    enriquecido = agregar_features_candidatas(df)
    X = enriquecido[columnas].copy()
    resumen = pd.read_csv(root / "models/resumen_reglas_extension_020.csv", float_precision="round_trip")
    resultados = pd.read_csv(root / "models/resultados_reglas_extension_020.csv", float_precision="round_trip")
    desarrollo = resumen.loc[resumen.regla.eq(REGLA_FINAL["nombre"])]
    final = resultados.loc[resultados.regla.eq(REGLA_FINAL["nombre"]) & resultados.anio_prueba.eq(2025)]
    historicos = resultados.loc[resultados.regla.eq(REGLA_FINAL["nombre"]) & resultados.anio_prueba.isin(ANIOS_DESARROLLO)]
    if len(desarrollo) != 1 or len(final) != 1 or sorted(historicos.anio_prueba.tolist()) != ANIOS_DESARROLLO:
        raise ValueError("Faltan las métricas exactas de desarrollo/comprobación de la regla final.")
    for metrica in bt.METRICAS:
        if not np.isclose(historicos[metrica].mean(), desarrollo.iloc[0][metrica+"_promedio"], atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError(f"Resumen inconsistente: {metrica}.")
    plan = {
        "algoritmo_final": "XGBoost", "conjunto_features": "D",
        "numero_features": len(columnas), "lista_features": columnas,
        "filas_entrenamiento_final": len(df),
        "periodos_entrada_usados": sorted(df.periodo_actual.unique().tolist()),
        "periodos_objetivo_usados": sorted(df.periodo_predicho.unique().tolist()),
        "conteo_clases_entrenamiento": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "dataset_sha256": sha, "version_dataset_d": "D:" + sha,
        "dataset_metadata_sha256": sha256_archivo(csv.parent / "dataset_metadata.json"),
        "version_politica_calidad": VERSION_POLITICA,
        "regla_decision": dict(REGLA_FINAL), "anios_desarrollo": ANIOS_DESARROLLO,
        "advertencia_2025": "comprobación adicional; no holdout virgen",
        "resultados_desarrollo": desarrollo.iloc[0].to_dict(),
        "resultados_comprobacion_2025": final.iloc[0].to_dict(),
        "hiperparametros": seleccion["hiperparametros"],
        "estrategia_desbalance": "sample_weight balanceado: N / (3 * N_clase), calculado sobre todo el entrenamiento final",
        "pesos_por_clase": {str(k): len(y)/(3*int(v)) for k, v in y.value_counts().sort_index().items()},
        "es_modelo_final_produccion": False,
    }
    return X, y.astype(int), normalizar(plan)


def evaluar_supera_baseline(metadata, root):
    """Compatibilidad: F1 macro > mejor baseline + 0.02, sin nueva evaluación.

    Conserva el margen existente en entrenar_modelo.MARGEN_F1_BASELINE.
    Usa la media de los cinco años de desarrollo, con igual peso por año.
    Nunca consulta métricas históricas anidadas, metricas_baseline.csv ni 2025.
    """
    root = Path(root)
    margen = 0.02
    evidencia = {
        "criterio": "F1 macro promedio de la regla final > mejor F1 macro promedio baseline + margen",
        "margen_f1_requerido": margen,
        "anios": list(ANIOS_DESARROLLO),
        "agregacion": "media aritmética por año; igual peso por año",
        "alcance": "evidencia temporal del pipeline final; no evaluación independiente del joblib reajustado",
        "evidencia_verificada": False,
        "fuentes_sha256": {},
    }
    try:
        nombres = ("comparacion_backtesting_temporal.json", "metricas_backtesting_temporal.csv",
                   "seleccion_regla_extension_020.json", "resultados_reglas_extension_020.csv")
        for nombre in nombres:
            ruta = "models/" + nombre
            huella = sha256_archivo(root / ruta)
            evidencia["fuentes_sha256"][ruta] = huella
            original = metadata.get("evidencia_preservada_sha256", {}).get(ruta)
            if original is not None and original != huella:
                raise ValueError(f"La evidencia cambió desde el entrenamiento: {ruta}.")
        comparacion = json.loads((root / "models" / nombres[0]).read_text(encoding="utf-8"))
        seleccion = json.loads((root / "models" / nombres[2]).read_text(encoding="utf-8"))
        if (metadata.get("conjunto_features") != "D"
                or metadata.get("algoritmo_final") != "XGBoost"
                or metadata.get("regla_decision") != REGLA_FINAL
                or seleccion.get("regla_seleccionada") != REGLA_FINAL["nombre"]
                or seleccion.get("anios_desarrollo") != ANIOS_DESARROLLO
                or seleccion.get("columnas_predictoras") != metadata.get("lista_features")
                or seleccion.get("estado") != "completado_sin_produccion"):
            raise ValueError("La evidencia no corresponde al pipeline XGBoost D y regla final.")
        sha = metadata.get("dataset_sha256")
        if (not sha or comparacion.get("dataset_sha256") != sha
                or seleccion.get("procedencia", {}).get("dataset_sha256") != sha):
            raise ValueError("Las evaluaciones no acreditan el mismo dataset.")
        base = pd.read_csv(root / "models" / nombres[1], float_precision="round_trip")
        reglas = pd.read_csv(root / "models" / nombres[3], float_precision="round_trip")
        final = reglas.loc[reglas.regla.eq(REGLA_FINAL["nombre"])
                           & reglas.anio_prueba.isin(ANIOS_DESARROLLO)].set_index("anio_prueba")
        if sorted(final.index.tolist()) != ANIOS_DESARROLLO:
            raise ValueError("Faltan folds únicos de desarrollo de la regla final.")
        medias = {}
        for nombre in ("Clase_Mayoritaria", "Persistencia_Riesgo_Actual", "Regla_Ocupacion_Actual"):
            baseline = base.loc[base.tipo.eq("baseline") & base.modelo.eq(nombre)
                                & base.anio_prueba.isin(ANIOS_DESARROLLO)].set_index("anio_prueba")
            if sorted(baseline.index.tolist()) != ANIOS_DESARROLLO:
                raise ValueError(f"Faltan folds únicos del baseline {nombre}.")
            for columna in ("test_sha256", "n_test", "n_train"):
                if not baseline.loc[ANIOS_DESARROLLO, columna].eq(final.loc[ANIOS_DESARROLLO, columna]).all():
                    raise ValueError(f"No coinciden registros/folds del baseline {nombre}.")
            if not np.isfinite(baseline.f1_macro).all() or not baseline.f1_macro.between(0, 1).all():
                raise ValueError(f"F1 inválido para {nombre}.")
            medias[nombre] = float(baseline.f1_macro.mean())
        if not np.isfinite(final.f1_macro).all() or not final.f1_macro.between(0, 1).all():
            raise ValueError("F1 de la regla final inválido.")
        f1 = float(final.f1_macro.mean())
        mejor = max(medias.values())
        supera = bool(f1 > mejor + margen)
        evidencia.update(evidencia_verificada=True, f1_macro_regla_final_promedio=f1,
            f1_macro_baselines_promedio=medias, mejor_f1_macro_baseline_promedio=mejor,
            mejora_f1_macro=f1-mejor,
            motivo=("La mejora supera estrictamente el margen requerido." if supera else
                    "La mejora no supera estrictamente 0.02 de F1 macro respecto al mejor baseline."))
        return supera, evidencia
    except (OSError, ValueError, KeyError, TypeError) as error:
        evidencia["motivo"] = f"No se puede demostrar superioridad con evidencia vigente: {error}"
        return False, evidencia


def actualizar_solo_metadata(root=ROOT):
    """Corrige solo dos campos de metadata; nunca carga ni ajusta el estimador."""
    root = Path(root)
    ruta = root / "models/model_metadata.json"
    original = ruta.read_bytes()
    metadata = json.loads(original.decode("utf-8"))
    modelo = root / "models/modelo_ipress.joblib"
    huella = sha256_archivo(modelo)
    if metadata.get("es_modelo_final_produccion") is not True or metadata.get("modelo_sha256") != huella:
        raise ValueError("Modelo final y metadata no coinciden; no se modifica nada.")
    metadata["supera_baseline"], metadata["comparacion_baselines_vigente"] = evaluar_supera_baseline(metadata, root)
    # No cambiar fechas de entrenamiento, código original, métricas ni SHA del joblib.
    contenido = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.TemporaryDirectory(prefix=".metadata_", dir=ruta.parent) as temporal:
        nueva = Path(temporal) / ruta.name
        nueva.write_text(contenido, encoding="utf-8")
        if ruta.read_bytes() != original or sha256_archivo(modelo) != huella:
            raise RuntimeError("Los artefactos cambiaron durante la corrección; operación cancelada.")
        os.replace(nueva, ruta)
    return metadata


def crear_metadata(plan, pipeline, root):
    """Solo se llama DESPUÉS de ajustar el pipeline; no inventa métricas nuevas."""
    from src.entrenar_modelo import _mapear_importancias_transformadas
    anterior_path = root / "models/model_metadata.json"
    anterior = json.loads(anterior_path.read_text(encoding="utf-8")) if anterior_path.exists() else {}
    fecha = datetime.now(timezone.utc).isoformat()
    importancias = _mapear_importancias_transformadas(pipeline, pipeline.named_steps["modelo"].feature_importances_)
    metadata = dict(plan)
    metadata.update({
        "tipo_modelo": "prediccion_siguiente_mes", "mejor_modelo": "XGBoost",
        "horizonte_prediccion": "mes_siguiente",
        "unidad_prediccion": "IPRESS + servicio hospitalario + mes",
        "variable_objetivo": bt.OBJETIVO,
        "definicion_objetivo": "Ocupación observada en t+1 exacto: bajo <0.70, medio >=0.70 y <0.85, alto >=0.85",
        "fecha_generacion": fecha, "fecha_entrenamiento": fecha,
        "columnas_predictoras_usadas": plan["lista_features"],
        "anios_disponibles_en_dataset": sorted({int(p[:4]) for p in plan["periodos_entrada_usados"]}),
        "definicion_probabilidad": SIGNIFICADO_PROBABILIDAD,
        "definicion_riesgo_insuficiencia_capacidad": SIGNIFICADO_INDICE,
        "importancia_variables_final": importancias.to_dict(orient="records"),
        "variables_mas_importantes": importancias.head(10).to_dict(orient="records"),
        "metricas_historicas_no_vigentes": anterior,
        "es_modelo_final_produccion": True,
        "advertencia_metodologica": (
            "Las métricas son evidencia temporal de pipelines ajustados por fold, no una evaluación "
            "independiente del artefacto final reajustado con todos los datos. Incluye 2025 y datos "
            "elegibles posteriores. No se reevalúa 2025 ni se usa para volver a elegir. "
            "El Dataset D omite meses sin pareja objetivo; no se reconstruye desde RAW. "
            "Los conteos de riesgo D usan la ocupación observada hasta t, nunca etiquetas futuras."
        ),
        "entorno": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                    for p in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")}},
        "codigo_sha256": {p: sha256_archivo(root / "src" / p) for p in (
            "entrenar_modelo_final.py", "entrenar_modelo.py", "modelo_final.py",
            "variables_temporales_experimentales.py", "variables_temporales.py")},
    })
    metadata["supera_baseline"], metadata["comparacion_baselines_vigente"] = evaluar_supera_baseline(metadata, root)
    return normalizar(metadata)


def publicar(modelo, metadata, root, entrada_verificacion):
    """Valida serialización antes de reemplazar; revierte ambos archivos ante error."""
    destino = root / "models"
    nombres = ("modelo_ipress.joblib", "model_metadata.json")
    anteriores = {n: (destino / n).read_bytes() if (destino / n).exists() else None for n in nombres}
    with tempfile.TemporaryDirectory(prefix=".final_", dir=destino) as temporal:
        temporal = Path(temporal)
        joblib.dump(modelo, temporal / nombres[0])
        recargado = joblib.load(temporal / nombres[0])
        if list(recargado.feature_names_in_) != metadata["lista_features"]:
            raise RuntimeError("El artefacto serializado no conserva las columnas D.")
        if not np.array_equal(modelo.predict_proba(entrada_verificacion),
                              recargado.predict_proba(entrada_verificacion)):
            raise RuntimeError("La serialización cambió las probabilidades del modelo.")
        metadata["modelo_sha256"] = sha256_archivo(temporal / nombres[0])
        (temporal / nombres[1]).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        try:
            for n in nombres:
                os.replace(temporal / n, destino / n)
        except Exception:
            for n, contenido in anteriores.items():
                if contenido is None:
                    (destino / n).unlink(missing_ok=True)
                else:
                    (destino / n).write_bytes(contenido)
            raise


def entrenar_final(root=ROOT):
    root = Path(root)
    huellas = huellas_protegidas(root)
    try:
        X, y, plan = leer_plan(root)
        print(f"Entrenamiento final: {len(X)} filas, {len(X.columns)} features D; un ajuste.", flush=True)
        from src import entrenar_modelo as motor
        columnas = [*motor.COLUMNAS_PREDICTORAS, *CONJUNTOS['D']]
        if columnas != plan["lista_features"]:
            raise ValueError("Las columnas actuales difieren del contrato D congelado.")
        algoritmo = motor.obtener_modelos().get("XGBoost")
        if algoritmo is None or normalizar(algoritmo.get_params()) != plan["hiperparametros"]:
            raise ValueError("XGBoost no está disponible o su configuración difiere del experimento.")
        clases = json.loads((root / "models/clases_riesgo.json").read_text(encoding="utf-8"))
        if clases != {"0": "bajo", "1": "medio", "2": "alto"}:
            raise ValueError("Mapeo de clases incompatible; no se sobrescribe automáticamente.")
        pipeline = motor.ajustar_pipeline("XGBoost", motor.crear_pipeline(X, algoritmo), X, y)
        modelo = ModeloFinalD(pipeline)
        modelo.predict_proba(X.iloc[:32])  # prueba del contrato, no medición de rendimiento
        metadata = crear_metadata(plan, pipeline, root)
        metadata["evidencia_preservada_sha256"] = huellas
        verificar_huellas(root, huellas)
        publicar(modelo, metadata, root, X.iloc[:32])
        print(json.dumps({k: metadata[k] for k in ("filas_entrenamiento_final", "numero_features", "modelo_sha256")}), flush=True)
        return metadata
    finally:
        verificar_huellas(root, huellas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    opciones = parser.add_mutually_exclusive_group()
    opciones.add_argument("--solo-plan", action="store_true", help="Validar D y evidencia sin entrenar ni escribir.")
    opciones.add_argument("--actualizar-solo-metadata", action="store_true",
                          help="Restaurar supera_baseline sin entrenar ni modificar el joblib.")
    args = parser.parse_args()
    if args.actualizar_solo_metadata:
        metadata = actualizar_solo_metadata()
        print(json.dumps({k: metadata[k] for k in ("supera_baseline", "comparacion_baselines_vigente")},
                         ensure_ascii=False, indent=2))
    elif args.solo_plan:
        _, _, plan = leer_plan()
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        entrenar_final()


if __name__ == "__main__":
    main()
