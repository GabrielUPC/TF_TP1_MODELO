"""Evaluación sellada 2025 del modelo binario; 2026 permanece cerrado."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analizar_calibracion_threshold import (
    tabla_calibracion, resumen_calibracion, metricas_threshold,
    validar_probabilidades,
)
from .backtesting_binario import (
    CONFIGURACION, OBJETIVO, VERSION_TARGET, calcular_metricas_binarias,
    crear_pipeline_binario, obtener_modelos_binarios, validar_salida_externa,
)
from .features_binarias import COLUMNAS_PREDICTORAS_BINARIAS, validar_predictoras
from .tratamiento_capacidad import VERSION_POLITICA

ANIO_HOLDOUT, ANIO_RESERVADO = 2025, 2026
THRESHOLDS = {"XGBoost": .36, "Random_Forest": .37, "Regresion_Logistica": .45}
SENSIBILIDAD_XGBOOST = [.32, .28]
SEMILLA_BOOTSTRAP, REPETICIONES_BOOTSTRAP = 20250905, 1000
ARCHIVOS_FINALES = [
    "metricas_holdout_2025.csv", "metricas_threshold_036_2025.csv",
    "sensibilidad_thresholds_2025.csv", "metricas_modelos_2025.csv",
    "metricas_familias_2025.csv", "calibracion_2025.csv",
    "intervalos_confianza_holdout_2025.csv", "drift_2025.csv",
    "predicciones_holdout_2025.csv", "verificacion_holdout_2025.json",
]


def _sha(path):
    with Path(path).open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def _metadata_valida(dataset_path, metadata_path):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    target, politica = metadata.get("definicion_target", {}), metadata.get("tratamiento_capacidad", {})
    if metadata.get("dataset_sha256") != _sha(dataset_path):
        raise ValueError("El hash del dataset no coincide con su metadata.")
    if (target.get("version") != VERSION_TARGET or target.get("variable_objetivo") != OBJETIVO
            or target.get("comparador") != ">" or target.get("umbral") != .90):
        raise ValueError("Metadata sin target binario estricto vigente.")
    if (politica.get("version") != VERSION_POLITICA
            or politica.get("reglas_aplicadas") != ["Q05", "Q06", "Q07", "Q08"]):
        raise ValueError("Metadata sin política Q05-Q08 vigente.")
    return metadata


def leer_sin_etiquetas(dataset_path):
    columnas = [*COLUMNAS_PREDICTORAS_BINARIAS, "codigo_ipress", "periodo_actual",
                "periodo_predicho", "familia_servicio"]
    disponibles = pd.read_csv(dataset_path, nrows=0).columns
    faltantes = sorted(set(columnas).difference(disponibles))
    if faltantes:
        raise ValueError("Faltan columnas para auditar holdout: " + ", ".join(faltantes))
    df = pd.read_csv(dataset_path, usecols=columnas, float_precision="round_trip",
                     dtype={"codigo_ipress": "string", "id_hospitalizacion": "string",
                            "ubigeo": "string"})
    validar_predictoras(COLUMNAS_PREDICTORAS_BINARIAS)
    return df


def particionar_y_auditar(df):
    actual = pd.PeriodIndex(df.periodo_actual.astype(str), freq="M")
    predicho = pd.PeriodIndex(df.periodo_predicho.astype(str), freq="M")
    if actual.hasnans or predicho.hasnans or not (predicho == actual + 1).all():
        raise ValueError("Existe salto temporal distinto de t a t+1.")
    train = predicho.year <= 2024
    test = predicho.year == ANIO_HOLDOUT
    reservado = predicho.year >= ANIO_RESERVADO
    if not train.any() or not test.any() or set(predicho[test].month) != set(range(1, 13)):
        raise ValueError("Train vacío o holdout 2025 incompleto.")
    claves = ["codigo_ipress", "servicio_hospitalizacion", "periodo_actual"]
    duplicados = int(df.duplicated(claves).sum())
    ids_train = set(map(tuple, df.loc[train, claves].astype(str).to_numpy()))
    ids_test = set(map(tuple, df.loc[test, claves].astype(str).to_numpy()))
    grupos = ["codigo_ipress", "servicio_hospitalizacion"]
    grupos_train = set(map(tuple, df.loc[train, grupos].astype(str).to_numpy()))
    grupos_test = set(map(tuple, df.loc[test, grupos].astype(str).to_numpy()))
    auditoria = {
        "estado": "AUDITORIA_PREAPERTURA_APROBADA",
        "etiquetas_2025_leidas": False, "metricas_2025_calculadas": False,
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "n_reservado_2026": int(reservado.sum()),
        "max_periodo_features_train": str(actual[train].max()),
        "max_periodo_predicho_train": str(predicho[train].max()),
        "min_periodo_features_test": str(actual[test].min()),
        "max_periodo_features_test": str(actual[test].max()),
        "min_periodo_predicho_test": str(predicho[test].min()),
        "max_periodo_predicho_test": str(predicho[test].max()),
        "train_solo_targets_hasta_2024": bool((predicho[train].year <= 2024).all()),
        "test_solo_targets_2025": bool((predicho[test].year == 2025).all()),
        "interseccion_observaciones_train_test": len(ids_train & ids_test),
        "grupos_ipress_servicio_compartidos": len(grupos_train & grupos_test),
        "duplicados_ipress_servicio_mes": duplicados,
        "saltos_temporales": int((predicho != actual + 1).sum()),
        "2026_excluido_train_test": bool(not (train & reservado).any() and not (test & reservado).any()),
    }
    if (duplicados or auditoria["interseccion_observaciones_train_test"]
            or not auditoria["train_solo_targets_hasta_2024"]
            or not auditoria["test_solo_targets_2025"]):
        raise ValueError("La auditoría temporal no permite abrir el holdout.")
    return train, test, reservado, auditoria


def _validar_decisiones(escenarios_path):
    tabla = pd.read_csv(escenarios_path)
    for modelo, threshold in THRESHOLDS.items():
        filas = tabla.loc[(tabla.modelo == modelo) & (tabla.escenario == "max_f1")]
        if len(filas) != 1 or not np.isclose(filas.threshold.iloc[0], threshold):
            raise ValueError("El threshold congelado no coincide con desarrollo.")
    return tabla


def preauditar(dataset_path, metadata_path, escenarios_path, probabilidades_desarrollo_path, output_dir):
    destino = validar_salida_externa(output_dir)
    if (destino / "auditoria_temporal_holdout_2025.json").exists():
        raise FileExistsError("La preauditoría sellada ya existe; no se sobrescribe.")
    _metadata_valida(dataset_path, metadata_path); _validar_decisiones(escenarios_path)
    desarrollo = pd.read_csv(probabilidades_desarrollo_path, float_precision="round_trip",
                              dtype={"codigo_ipress": "string"})
    if pd.to_numeric(desarrollo.anio_prueba, errors="raise").ge(2025).any():
        raise ValueError("La fuente de desarrollo contiene un periodo reservado.")
    validar_probabilidades(desarrollo.loc[desarrollo.tipo_score.eq("probabilidad_sin_calibracion")].copy())
    base = leer_sin_etiquetas(dataset_path)
    _, _, _, auditoria = particionar_y_auditar(base)
    destino.mkdir(parents=True, exist_ok=True)
    congelacion = {
        "modelo_principal": "XGBoost", "threshold_principal": .36,
        "thresholds_sensibilidad": [.32, .28], "challenger": "Random_Forest",
        "benchmark": "Regresion_Logistica", "thresholds_challengers": {
            "Random_Forest": .37, "Regresion_Logistica": .45},
        "configuraciones": CONFIGURACION, "features": COLUMNAS_PREDICTORAS_BINARIAS,
        "seleccion_realizada_con": "2018-2024", "2025_usado_para_decisiones": False,
        "2026_usado": False, "dataset_sha256": _sha(dataset_path),
        "metadata_sha256": _sha(metadata_path), "escenarios_desarrollo_sha256": _sha(escenarios_path),
        "probabilidades_desarrollo_sha256": _sha(probabilidades_desarrollo_path),
    }
    congelacion_path = destino / "congelacion_preholdout_2025.json"
    congelacion_path.write_text(json.dumps(congelacion, ensure_ascii=False, indent=2), encoding="utf-8")
    auditoria["dataset_sha256"] = congelacion["dataset_sha256"]
    auditoria["congelacion_sha256"] = _sha(congelacion_path)
    (destino / "auditoria_temporal_holdout_2025.json").write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    return auditoria


def _leer_filas(dataset_path, posiciones, columnas):
    permitidas = set((np.asarray(posiciones, dtype=int) + 1).tolist())
    return pd.read_csv(dataset_path, usecols=columnas, float_precision="round_trip",
                       skiprows=lambda numero: numero > 0 and numero not in permitidas)


def _features(df):
    X = df[COLUMNAS_PREDICTORAS_BINARIAS].copy()
    for columna in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[columna] = X[columna].astype(object).where(X[columna].notna(), np.nan)
    return X


def bootstrap_cluster(y, score, clusters, threshold=.36, repeticiones=REPETICIONES_BOOTSTRAP,
                      semilla=SEMILLA_BOOTSTRAP):
    y, score, clusters = np.asarray(y, int), np.asarray(score, float), np.asarray(clusters)
    grupos = [np.flatnonzero(clusters == valor) for valor in np.unique(clusters)]
    rng = np.random.default_rng(semilla)
    nombres = ["roc_auc", "average_precision", "recall", "especificidad",
               "precision_ppv", "f1", "brier_score"]
    valores = {n: [] for n in nombres}
    for _ in range(repeticiones):
        indices = np.concatenate([grupos[i] for i in rng.integers(0, len(grupos), size=len(grupos))])
        raw = calcular_metricas_binarias(y[indices], score[indices])
        dec = metricas_threshold(y[indices], score[indices], threshold)
        calculadas = {"roc_auc": raw["roc_auc"], "average_precision": raw["average_precision"],
                      "brier_score": raw["brier_score"], **dec}
        for nombre in nombres: valores[nombre].append(calculadas[nombre])
    raw, dec = calcular_metricas_binarias(y, score), metricas_threshold(y, score, threshold)
    metricas_independientes = {"roc_auc", "average_precision", "brier_score"}
    filas = []
    for nombre in nombres:
        validos = np.asarray(valores[nombre], float); validos = validos[np.isfinite(validos)]
        estimacion = raw[nombre] if nombre in metricas_independientes else dec[nombre]
        filas.append({"metrica": nombre, "estimacion": estimacion,
                      "ic95_inferior": np.quantile(validos, .025),
                      "ic95_superior": np.quantile(validos, .975),
                      "repeticiones_validas": len(validos), "semilla": semilla,
                      "metodo": "bootstrap por cluster IPRESS+servicio"})
    return pd.DataFrame(filas)


def _psi(referencia, actual):
    referencia, actual = np.asarray(referencia, float), np.asarray(actual, float)
    cortes = np.unique(np.quantile(referencia[np.isfinite(referencia)], np.linspace(0, 1, 11)))
    if len(cortes) < 3: return np.nan
    cortes[0], cortes[-1] = -np.inf, np.inf
    a, b = np.histogram(referencia, bins=cortes)[0]/len(referencia), np.histogram(actual, bins=cortes)[0]/len(actual)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b-a)*np.log(b/a)))


def _drift(base, train, test, desarrollo, scores, y_test, metricas_modelos):
    from scipy.stats import ks_2samp
    filas = []
    prevalencia_dev = desarrollo.drop_duplicates(["anio_prueba", "indice_dataset"]).y_real.mean()
    filas.append({"categoria": "prevalencia", "variable": OBJETIVO,
                  "desarrollo": prevalencia_dev, "holdout_2025": y_test.mean(),
                  "diferencia": y_test.mean()-prevalencia_dev})
    for modelo, score in scores.items():
        previo = desarrollo.loc[desarrollo.modelo.eq(modelo), "score_positivo"].to_numpy(float)
        ks = ks_2samp(previo, score)
        filas.append({"categoria": "score", "variable": modelo, "desarrollo": previo.mean(),
                      "holdout_2025": score.mean(), "diferencia": score.mean()-previo.mean(),
                      "psi": _psi(previo, score), "ks": ks.statistic, "ks_pvalue": ks.pvalue})
    for columna in base[COLUMNAS_PREDICTORAS_BINARIAS].select_dtypes(include="number"):
        a = pd.to_numeric(base.loc[train, columna], errors="coerce").to_numpy(float)
        b = pd.to_numeric(base.loc[test, columna], errors="coerce").to_numpy(float)
        af, bf = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(af) or not len(bf): continue
        sd = np.sqrt((np.var(af)+np.var(bf))/2)
        filas.append({"categoria": "feature_numerica", "variable": columna,
                      "desarrollo": np.mean(af), "holdout_2025": np.mean(bf),
                      "diferencia": np.mean(bf)-np.mean(af), "smd": (np.mean(bf)-np.mean(af))/sd if sd else 0,
                      "psi": _psi(af, bf), "missing_desarrollo": 1-len(af)/len(a),
                      "missing_2025": 1-len(bf)/len(b)})
    for _, fila in metricas_modelos.iterrows():
        previo = desarrollo.loc[desarrollo.modelo.eq(fila.modelo)]
        raw = calcular_metricas_binarias(previo.y_real, previo.score_positivo)
        for metrica in ["roc_auc", "average_precision", "brier_score"]:
            filas.append({"categoria": "performance", "variable": f"{fila.modelo}:{metrica}",
                          "desarrollo": raw[metrica], "holdout_2025": fila[metrica],
                          "diferencia": fila[metrica]-raw[metrica]})
    return pd.DataFrame(filas)


def evaluar(dataset_path, metadata_path, escenarios_path, probabilidades_desarrollo_path,
            output_dir, *, modelos=None, constructor_pipeline=None,
            repeticiones_bootstrap=REPETICIONES_BOOTSTRAP):
    destino = validar_salida_externa(output_dir)
    auditoria_path, congelacion_path = destino/"auditoria_temporal_holdout_2025.json", destino/"congelacion_preholdout_2025.json"
    if not auditoria_path.exists() or not congelacion_path.exists():
        raise ValueError("Debe completarse la preauditoría antes de abrir 2025.")
    if any((destino/n).exists() for n in ARCHIVOS_FINALES):
        raise FileExistsError("El holdout ya fue abierto; no se reevalúa.")
    auditoria = json.loads(auditoria_path.read_text(encoding="utf-8")); congelacion = json.loads(congelacion_path.read_text(encoding="utf-8"))
    if (auditoria.get("estado") != "AUDITORIA_PREAPERTURA_APROBADA"
            or congelacion.get("dataset_sha256") != _sha(dataset_path)
            or auditoria.get("congelacion_sha256") != _sha(congelacion_path)):
        raise ValueError("La evidencia preapertura no coincide con las fuentes.")
    _metadata_valida(dataset_path, metadata_path); _validar_decisiones(escenarios_path)
    desarrollo = pd.read_csv(probabilidades_desarrollo_path, float_precision="round_trip", dtype={"codigo_ipress": "string"})
    desarrollo = validar_probabilidades(desarrollo.loc[desarrollo.tipo_score.eq("probabilidad_sin_calibracion")].copy())
    base = leer_sin_etiquetas(dataset_path); train, test, reservado, nueva = particionar_y_auditar(base)
    if nueva["n_train"] != auditoria["n_train"] or nueva["n_test"] != auditoria["n_test"]:
        raise ValueError("La partición cambió después de la preauditoría.")
    columnas_y = ["periodo_predicho", OBJETIVO, "ocupacion_siguiente_mes"]
    y_train_df = _leer_filas(dataset_path, np.flatnonzero(train), columnas_y)
    y_train = pd.to_numeric(y_train_df[OBJETIVO], errors="raise").to_numpy(int)
    ocupacion_train = pd.to_numeric(y_train_df.ocupacion_siguiente_mes, errors="raise").to_numpy(float)
    if not np.array_equal(y_train, (ocupacion_train > .90).astype(int)):
        raise ValueError("Target de train incompatible con ocupación t+1.")
    X_train, X_test = _features(base.loc[train]), _features(base.loc[test])
    modelos = obtener_modelos_binarios() if modelos is None else modelos
    if set(modelos) != set(CONFIGURACION): raise ValueError("Se requieren los tres modelos congelados.")
    constructor = constructor_pipeline or crear_pipeline_binario
    scores = {}
    for nombre, algoritmo in modelos.items():
        pipeline = constructor(X_train, algoritmo); pipeline.fit(X_train, y_train)
        proba = np.asarray(pipeline.predict_proba(X_test), float); clases = list(pipeline.classes_)
        if proba.shape != (len(X_test), 2) or set(clases) != {0, 1}: raise ValueError("predict_proba inválida.")
        scores[nombre] = proba[:, clases.index(1)]
    pre = {"predicciones_generadas_antes_de_abrir_etiquetas_2025": True, "n_predicciones": len(X_test),
           "sha256_scores": {m: hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest() for m,s in scores.items()}}
    (destino/"predicciones_preapertura_2025.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    y_test_df = _leer_filas(dataset_path, np.flatnonzero(test), columnas_y)
    y_test = pd.to_numeric(y_test_df[OBJETIVO], errors="raise").to_numpy(int)
    ocupacion_test = pd.to_numeric(y_test_df.ocupacion_siguiente_mes, errors="raise").to_numpy(float)
    if not np.array_equal(y_test, (ocupacion_test > .90).astype(int)): raise ValueError("Target 2025 incompatible.")
    auditoria.update({"etiquetas_2025_leidas": True, "aperturas_etiqueta_2025": 1,
                      "ajuste_terminado_antes_de_abrir_etiquetas": True,
                      "positivos_test": int(y_test.sum()), "negativos_test": int(len(y_test)-y_test.sum()),
                      "prevalencia_test": float(y_test.mean())})
    auditoria_path.write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = base.loc[test, ["codigo_ipress","servicio_hospitalizacion","periodo_actual","periodo_predicho","familia_servicio"]].reset_index(drop=True)
    predicciones=[]
    for modelo,score in scores.items():
        p=ids.copy();p["modelo"]=modelo;p["y_real"]=y_test;p["score_positivo"]=score;predicciones.append(p)
    predicciones=pd.concat(predicciones,ignore_index=True)
    cal=tabla_calibracion(predicciones,("modelo",)); cal_resumen=resumen_calibracion(cal).set_index("modelo")
    filas=[]
    for modelo,score in scores.items():
        raw=calcular_metricas_binarias(y_test,score);dec=metricas_threshold(y_test,score,THRESHOLDS[modelo])
        filas.append({"modelo":modelo,"threshold_congelado":THRESHOLDS[modelo],**raw,
                      **{f"decision_{k}":v for k,v in dec.items()},"ece_10_bins":cal_resumen.loc[modelo,"ece_10_bins"],
                      "sesgo_calibracion":cal_resumen.loc[modelo,"sesgo_calibracion"],"probabilidad_media":float(np.mean(score))})
    metricas_modelos=pd.DataFrame(filas);xgb=metricas_modelos.query("modelo=='XGBoost'")
    xgb[["modelo","n","positivos","negativos","roc_auc","average_precision","brier_score","ece_10_bins","sesgo_calibracion","prevalencia_real","probabilidad_media"]].to_csv(destino/"metricas_holdout_2025.csv",index=False)
    principal=metricas_threshold(y_test,scores["XGBoost"],.36)
    pd.DataFrame([{"modelo":"XGBoost","escenario":"PRINCIPAL_PREESPECIFICADO",**principal}]).to_csv(destino/"metricas_threshold_036_2025.csv",index=False)
    sensibilidad=[{"modelo":"XGBoost","tipo":tipo,**metricas_threshold(y_test,scores["XGBoost"],th)} for th,tipo in [(.36,"principal"),(.32,"sensibilidad"),(.28,"sensibilidad")]]
    pd.DataFrame(sensibilidad).to_csv(destino/"sensibilidad_thresholds_2025.csv",index=False);metricas_modelos.to_csv(destino/"metricas_modelos_2025.csv",index=False)
    familias=[]
    for familia,g in predicciones.query("modelo=='XGBoost'").groupby("familia_servicio"):
        raw=calcular_metricas_binarias(g.y_real,g.score_positivo);dec=metricas_threshold(g.y_real,g.score_positivo,.36)
        cf=tabla_calibracion(g.assign(modelo="XGBoost"),("modelo",));rc=resumen_calibracion(cf).iloc[0]
        familias.append({"modelo":"XGBoost","threshold":.36,"familia_servicio":familia,**raw,
                         **{f"decision_{k}":v for k,v in dec.items()},"ece_10_bins":rc.ece_10_bins})
    pd.DataFrame(familias).to_csv(destino/"metricas_familias_2025.csv",index=False)
    cal["alcance"]="global";cf=tabla_calibracion(predicciones.query("modelo=='XGBoost'"),("modelo","familia_servicio"));cf["alcance"]="familia"
    pd.concat([cal,cf],ignore_index=True).to_csv(destino/"calibracion_2025.csv",index=False)
    clusters=ids.codigo_ipress.astype(str)+"|"+ids.servicio_hospitalizacion.astype(str)
    bootstrap_cluster(y_test,scores["XGBoost"],clusters,.36,repeticiones_bootstrap).to_csv(destino/"intervalos_confianza_holdout_2025.csv",index=False)
    _drift(base,train,test,desarrollo,scores,y_test,metricas_modelos).to_csv(destino/"drift_2025.csv",index=False)
    predicciones.to_csv(destino/"predicciones_holdout_2025.csv",index=False)
    ver={"auditoria_preapertura_aprobada":True,"predicciones_antes_de_etiquetas":True,"aperturas_etiqueta_2025":1,
         "train_max_periodo_predicho":auditoria["max_periodo_predicho_train"],"test_min_periodo_predicho":auditoria["min_periodo_predicho_test"],
         "test_max_periodo_predicho":auditoria["max_periodo_predicho_test"],"n_train":auditoria["n_train"],"n_test":auditoria["n_test"],
         "2026_usado_train":False,"2026_predicho":False,"2026_evaluado":False,"modelo_threshold_reseleccionados_con_2025":False,
         "estimadores_productivos_guardados":False,"bootstrap":{"metodo":"clusters IPRESS+servicio","semilla":SEMILLA_BOOTSTRAP,"repeticiones":repeticiones_bootstrap},
         "archivos":{n:_sha(destino/n) for n in ARCHIVOS_FINALES[:-1]}}
    (destino/"verificacion_holdout_2025.json").write_text(json.dumps(ver,ensure_ascii=False,indent=2),encoding="utf-8")
    auditoria["metricas_2025_calculadas"] = True
    auditoria_path.write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    return metricas_modelos,pd.DataFrame(sensibilidad),pd.DataFrame(familias)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for nombre in ["dataset","metadata","escenarios-desarrollo","probabilidades-desarrollo","output-dir"]:p.add_argument("--"+nombre,type=Path,required=True)
    p.add_argument("--modo",choices=["preauditar","evaluar"],required=True);a=p.parse_args(argv)
    comunes=(a.dataset,a.metadata,a.escenarios_desarrollo,a.probabilidades_desarrollo,a.output_dir)
    print(json.dumps(preauditar(*comunes),ensure_ascii=False,indent=2)) if a.modo=="preauditar" else evaluar(*comunes)
    return 0


if __name__=="__main__":raise SystemExit(main())
