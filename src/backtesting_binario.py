"""Evaluacion binaria temporal de desarrollo; 2025 y posteriores quedan sellados.

No guarda estimadores ni invoca entrenamiento productivo. Los baselines emiten
scores deterministas 0/1, explicitamente distintos de probabilidades calibradas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features_binarias import (
    COLUMNAS_PREDICTORAS_BINARIAS, auditar_features, derivar_familia_servicio,
    validar_predictoras,
)
from .tratamiento_capacidad import VERSION_POLITICA

ROOT = Path(__file__).resolve().parent.parent
OBJETIVO = "alta_presion_capacidad_siguiente_mes"
VERSION_TARGET = "alta_presion_capacidad_t1_v1"
ANIO_RESERVADO = 2025
UMBRAL_TECNICO = 0.50
SEMILLA = 42
N_JOBS = 4
METRICAS = [
    "roc_auc", "average_precision", "recall", "specificity", "precision", "f1",
    "balanced_accuracy", "accuracy", "brier_score", "tp", "tn", "fp", "fn",
    "prevalencia_real", "prevalencia_predicha", "tasa_falsos_negativos",
]
ORDEN_RANKING = [
    "average_precision_promedio", "roc_auc_promedio", "average_precision_std",
    "recall_promedio", "brier_score_promedio", "min_average_precision_familia",
    "tasa_falsos_negativos_promedio",
]
SENTIDOS_RANKING = [False, False, True, False, True, False, True]
CONFIGURACION = {
    "Regresion_Logistica": {"max_iter": 2000, "random_state": SEMILLA},
    "Random_Forest": {"n_estimators": 300, "max_depth": 12,
                      "random_state": SEMILLA, "n_jobs": N_JOBS},
    "XGBoost": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": "binary:logistic", "eval_metric": "logloss",
                "random_state": SEMILLA, "n_jobs": N_JOBS},
}


def _clases(valores):
    datos = np.asarray(valores)
    if datos.ndim != 1 or not np.isin(datos, [0, 1]).all():
        raise ValueError("Se requieren etiquetas binarias 0/1 sin ausentes.")
    return datos.astype(int)


def validar_periodos(df):
    if df.empty or not df.index.is_unique:
        raise ValueError("Dataset vacio o indices duplicados.")
    actual = pd.PeriodIndex(df["periodo_actual"].astype(str), freq="M")
    objetivo = pd.PeriodIndex(df["periodo_predicho"].astype(str), freq="M")
    if actual.hasnans or objetivo.hasnans or not (objetivo == actual + 1).all():
        raise ValueError("Cada objetivo debe corresponder exactamente a t+1 calendario.")
    claves = ["codigo_ipress", "servicio_hospitalizacion", "periodo_actual"]
    if df[claves].isna().any().any() or df.duplicated(claves).any():
        raise ValueError("Claves ausentes o IPRESS-servicio-mes duplicados.")
    # La planificacion no inspecciona las clases de 2025/2026.
    _clases(df.loc[objetivo.year < ANIO_RESERVADO, OBJETIVO])
    return actual, objetivo


def validar_target_desarrollo(df):
    """Verifica la verdad del evento en toda entrada, tambien mediante API.

    Las observaciones reservadas de 2025/2026 no se inspeccionan.
    """
    _, periodos = validar_periodos(df)
    if "ocupacion_siguiente_mes" not in df:
        raise ValueError("Falta ocupacion_siguiente_mes para verificar el target.")
    desarrollo = df.loc[periodos.year < ANIO_RESERVADO]
    futuro = pd.to_numeric(desarrollo["ocupacion_siguiente_mes"], errors="raise").to_numpy(
        dtype=float, na_value=np.nan
    )
    if (not np.isfinite(futuro).all() or (futuro < 0).any()
            or not np.array_equal((futuro > .90).astype(int), _clases(desarrollo[OBJETIVO]))):
        raise ValueError("El target de desarrollo no coincide con ocupacion t+1 valida y >0.90.")


def planificar_folds_binarios(df, min_meses_historial=24, min_casos_clase=2):
    """24 periodos distintos anteriores; los huecos globales no invalidan.

    Para los anios reservados solo se informa existencia y n_test. Ninguna
    etiqueta, distribucion ni metrica predictiva reservada se consulta.
    """
    if min_meses_historial < 24 or min_casos_clase < 2:
        raise ValueError("Minimos: 24 periodos previos y 2 casos de cada clase.")
    _, periodos = validar_periodos(df)
    filas = []
    for anio in sorted(set(periodos.year)):
        test = periodos.year == anio
        if anio >= ANIO_RESERVADO:
            filas.append({"anio_prueba": int(anio), "elegible": False,
                          "motivo": "Reservado: solo existencia y cantidad de registros",
                          "n_test": int(test.sum()), "n_train": None,
                          "meses_train": None, "meses_test": None,
                          "train_desde": "", "train_hasta": "",
                          "train_clase_0": None, "train_clase_1": None,
                          "huecos_historial": None})
            continue
        train = periodos.year < anio
        meses = periodos[train].unique()
        conteos = df.loc[train, OBJETIVO].value_counts()
        motivos = []
        if set(periodos[test].month) != set(range(1, 13)):
            motivos.append("Anio objetivo incompleto")
        if len(meses) < min_meses_historial:
            motivos.append(f"Se requieren {min_meses_historial} periodos mensuales distintos anteriores")
        if any(conteos.get(c, 0) < min_casos_clase for c in (0, 1)):
            motivos.append(f"Se requieren {min_casos_clase} casos de cada clase en train")
        huecos = len(pd.period_range(meses.min(), meses.max(), freq="M"))-len(meses) if len(meses) else 0
        filas.append({"anio_prueba": int(anio), "elegible": not motivos,
                      "motivo": "; ".join(motivos) or "Cumple criterios",
                      "n_train": int(train.sum()), "n_test": int(test.sum()),
                      "meses_train": len(meses), "meses_test": len(periodos[test].unique()),
                      "train_desde": str(meses.min()) if len(meses) else "",
                      "train_hasta": str(meses.max()) if len(meses) else "",
                      "train_clase_0": int(conteos.get(0, 0)),
                      "train_clase_1": int(conteos.get(1, 0)),
                      "huecos_historial": huecos})
    return pd.DataFrame(filas)


def crear_folds_binarios(df, min_meses_historial=24, min_casos_clase=2):
    plan = planificar_folds_binarios(df, min_meses_historial, min_casos_clase)
    _, periodos = validar_periodos(df)
    folds = []
    for anio in plan.loc[plan.elegible, "anio_prueba"]:
        if anio >= ANIO_RESERVADO:
            raise ValueError("La evaluacion de 2025 o posteriores esta prohibida.")
        train, test = df.index[periodos.year < anio], df.index[periodos.year == anio]
        folds.append((int(anio), train, test))
    return folds, plan


def calcular_metricas_binarias(y_real, score_positivo, umbral=UMBRAL_TECNICO):
    """AP escalonada (no area trapezoidal); ties agrupados como sklearn.

    ROC-AUC requiere ambas clases. AP y recall son NaN sin positivos;
    especificidad es NaN sin negativos. Precision y F1 usan cero si su
    denominador es cero. El umbral .50 es tecnico provisional.
    """
    real = _clases(y_real)
    score = np.asarray(score_positivo, dtype=float)
    if (not len(real) or score.shape != real.shape or not np.isfinite(score).all()
            or (score < 0).any() or (score > 1).any() or umbral != UMBRAL_TECNICO):
        raise ValueError("Scores invalidos o umbral distinto del tecnico provisional 0.50.")
    pred = score >= umbral
    pos, neg = int(real.sum()), int(len(real)-real.sum())
    tp, tn = int(((real == 1) & pred).sum()), int(((real == 0) & ~pred).sum())
    fp, fn = neg-tn, pos-tp
    recall = tp/pos if pos else np.nan
    specificity = tn/neg if neg else np.nan
    precision = tp/(tp+fp) if tp+fp else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    auc = np.nan
    if pos and neg:
        rangos = pd.Series(score).rank(method="average").to_numpy()
        auc = (rangos[real == 1].sum()-pos*(pos+1)/2)/(pos*neg)
    ap = np.nan
    if pos:
        orden = np.argsort(-score, kind="stable")
        y, s = real[orden], score[orden]
        finales = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
        acumulados = np.cumsum(y)[finales]
        incrementos = np.diff(np.r_[0, acumulados])
        ap = np.sum(incrementos/pos * acumulados/(finales+1))
    return {"n": len(real), "positivos": pos, "negativos": neg,
            "roc_auc": float(auc), "average_precision": float(ap),
            "recall": float(recall), "specificity": float(specificity),
            "precision": float(precision), "f1": float(f1),
            "balanced_accuracy": float(np.nanmean([recall, specificity])),
            "accuracy": (tp+tn)/len(real), "brier_score": float(np.mean((score-real)**2)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "prevalencia_real": pos/len(real), "prevalencia_predicha": float(pred.mean()),
            "tasa_falsos_negativos": fn/pos if pos else np.nan}


def tabla_calibracion(y_real, score_positivo):
    real, score = _clases(y_real), np.asarray(score_positivo, dtype=float)
    calcular_metricas_binarias(real, score)
    bins = np.minimum((score*10).astype(int), 9)
    filas = []
    for intervalo in range(10):
        mascara = bins == intervalo
        filas.append({"bin": intervalo, "limite_inferior": intervalo/10,
                      "limite_superior": (intervalo+1)/10, "n": int(mascara.sum()),
                      "probabilidad_media": float(score[mascara].mean()) if mascara.any() else np.nan,
                      "frecuencia_observada": float(real[mascara].mean()) if mascara.any() else np.nan})
    return pd.DataFrame(filas)


def obtener_modelos_binarios():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    # Sin class_weight, scale_pos_weight, sample_weight ni remuestreo.
    return {"Regresion_Logistica": LogisticRegression(**CONFIGURACION["Regresion_Logistica"]),
            "Random_Forest": RandomForestClassifier(**CONFIGURACION["Random_Forest"]),
            "XGBoost": XGBClassifier(**CONFIGURACION["XGBoost"])}


def crear_pipeline_binario(X_train, algoritmo):
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    categoricas = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numericas = [c for c in X_train.columns if c not in categoricas]
    numerico = Pipeline([("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("escalar", StandardScaler())])
    categorico = Pipeline([("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                           ("codificar", OneHotEncoder(handle_unknown="ignore"))])
    preproceso = ColumnTransformer([("numericas", numerico, numericas),
                                   ("categoricas", categorico, categoricas)])
    return Pipeline([("preprocesamiento", preproceso), ("modelo", clone(algoritmo))])


def _features(df, columnas):
    X = df.loc[:, columnas].copy()
    for columna in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[columna] = X[columna].astype(object).where(X[columna].notna(), np.nan)
    numericas = X.select_dtypes(include="number")
    if np.isinf(numericas.to_numpy(dtype=float)).any():
        raise ValueError("Predictores con infinito; revisar preparacion antes de evaluar.")
    return X


def _sha256(path):
    with Path(path).open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def validar_salida_externa(output_dir):
    destino = Path(output_dir).resolve()
    if destino == ROOT.resolve() or ROOT.resolve() in destino.parents:
        raise ValueError("Los artefactos experimentales deben guardarse fuera del repositorio.")
    return destino


def resumir_modelos_binarios(metricas, estratos):
    if metricas.empty or metricas.duplicated(["modelo", "anio_prueba"]).any():
        raise ValueError("Resultados vacios o duplicados por modelo y anio.")
    conjuntos = metricas.groupby("modelo").anio_prueba.agg(lambda x: frozenset(x))
    if len(set(conjuntos)) != 1:
        raise ValueError("Modelos evaluados en anios distintos.")
    for _, grupo in metricas.groupby("anio_prueba"):
        if grupo.n_test.nunique() != 1 or grupo.test_sha256.nunique() != 1:
            raise ValueError("Todos los modelos y baselines deben usar el mismo test.")
    filas = []
    for nombre, grupo in metricas.groupby("modelo", sort=True):
        fila = {"modelo": nombre, "tipo": grupo.tipo.iloc[0], "n_anios": len(grupo)}
        for metrica in METRICAS:
            v = grupo[metrica]
            fila.update({f"{metrica}_promedio": v.mean(), f"{metrica}_std": v.std(ddof=0),
                         f"{metrica}_min": v.min(), f"{metrica}_max": v.max(),
                         f"{metrica}_n_validos": int(v.notna().sum())})
        por_familia = estratos.loc[estratos.modelo == nombre].groupby("familia_servicio").average_precision.mean()
        fila["min_average_precision_familia"] = (
            por_familia.min() if len(por_familia) == 2 and por_familia.notna().all() else np.nan
        )
        filas.append(fila)
    resumen = pd.DataFrame(filas).sort_values(ORDEN_RANKING, ascending=SENTIDOS_RANKING,
                                             na_position="last", kind="stable").reset_index(drop=True)
    resumen["ranking_candidato"] = pd.Series(pd.NA, index=resumen.index, dtype="Int64")
    candidatos = resumen.index[resumen.tipo.eq("modelo")]
    resumen.loc[candidatos, "ranking_candidato"] = range(1, len(candidatos)+1)
    return resumen


def ejecutar_backtesting_binario(df, output_dir, *, modelos=None, constructor_pipeline=None,
                                  min_meses_historial=24, min_casos_clase=2, procedencia=None):
    """Ajusta una vez cada algoritmo por fold exclusivamente anterior a 2025.

    Las features ya deben estar calculadas sobre los meses completos antes del
    recorte t+1; nunca reconstruye ventanas sobre el dataset de pares.
    """
    destino = validar_salida_externa(output_dir)
    columnas = validar_predictoras(COLUMNAS_PREDICTORAS_BINARIAS)
    faltantes = sorted(set(columnas).difference(df.columns))
    if faltantes:
        raise ValueError("Faltan features preparadas: " + ", ".join(faltantes))
    validar_target_desarrollo(df)
    folds, plan = crear_folds_binarios(df, min_meses_historial, min_casos_clase)
    if not folds:
        raise ValueError("Ningun anio de desarrollo elegible.")
    modelos = obtener_modelos_binarios() if modelos is None else modelos
    if set(modelos) != set(CONFIGURACION):
        raise ValueError("La comparacion requiere exactamente Logistic, Random Forest y XGBoost.")
    constructor = constructor_pipeline or crear_pipeline_binario
    destino.mkdir(parents=True, exist_ok=True)
    criterio = {"target": OBJETIVO, "version_target": VERSION_TARGET,
                "anios_desarrollo": [a for a, _, _ in folds], "holdout_reservado": 2025,
                "holdout_evaluado": False, "2026_utilizado": False,
                "orden_ranking_lexicografico": ORDEN_RANKING,
                "ascendente": SENTIDOS_RANKING, "std_ddof": 0,
                "umbral_tecnico_provisional": UMBRAL_TECNICO,
                "umbral_operacional_definitivo": False,
                "algoritmos_configuracion": CONFIGURACION,
                "features": columnas, "class_weight": None, "sample_weight": None,
                "remuestreo": False, "calibracion_aplicada": False,
                "baselines_score": "0/1 determinista; no es probabilidad calibrada",
                "ranking_incluye_baselines": False, "modelo_final_produccion": False,
                "procedencia": procedencia or {}}
    # Se fija el criterio antes del primer ajuste y antes de ver resultados.
    (destino/"criterio_comparacion_binario.json").write_text(json.dumps(criterio, indent=2, ensure_ascii=False), encoding="utf-8")
    plan.to_csv(destino/"plan_folds_binario.csv", index=False)
    auditar_features(df).to_csv(destino/"auditoria_features.csv", index=False)
    resultados, estratos, probas, calibracion = [], [], [], []
    for anio, train, test in folds:
        X_train, X_test = _features(df.loc[train], columnas), _features(df.loc[test], columnas)
        y_train, y_test = _clases(df.loc[train, OBJETIVO]), _clases(df.loc[test, OBJETIVO])
        familia = derivar_familia_servicio(df.loc[test, "id_hospitalizacion"])
        identificadores = df.loc[test, ["codigo_ipress", "servicio_hospitalizacion", "periodo_actual", "periodo_predicho"]].copy()
        huella = hashlib.sha256(pd.util.hash_pandas_object(identificadores, index=True).to_numpy().tobytes()).hexdigest()

        def registrar(nombre, tipo, score):
            metricas = calcular_metricas_binarias(y_test, score)
            tipo_score = "probabilidad_sin_calibracion" if tipo == "modelo" else "score_determinista_0_1"
            comun = {"modelo": nombre, "tipo": tipo, "anio_prueba": anio,
                     "n_train": len(train), "n_test": len(test), "test_sha256": huella,
                     "tipo_score": tipo_score, "umbral_tecnico_provisional": UMBRAL_TECNICO}
            resultados.append({**comun, **metricas})
            for valor in ("hospitalizacion", "cuidados_criticos"):
                mascara = familia.eq(valor).to_numpy(dtype=bool)
                if mascara.any():
                    m = calcular_metricas_binarias(y_test[mascara], np.asarray(score)[mascara])
                else:
                    m = {**{c: np.nan for c in METRICAS}, "n": 0, "positivos": 0, "negativos": 0}
                estratos.append({**comun, "familia_servicio": valor, **m})
            p = identificadores.copy()
            p["indice_dataset"] = test
            p["modelo"], p["tipo_score"], p["anio_prueba"] = nombre, tipo_score, anio
            p["familia_servicio"], p["y_real"] = familia.to_numpy(), y_test
            p["score_positivo"], p["y_pred"] = score, (np.asarray(score) >= UMBRAL_TECNICO).astype(int)
            probas.append(p)
            c = tabla_calibracion(y_test, score)
            for k, v in {"modelo": nombre, "tipo_score": tipo_score, "anio_prueba": anio}.items():
                c[k] = v
            calibracion.append(c)

        for nombre, algoritmo in modelos.items():
            print(f"Desarrollo binario {anio}: {nombre}", flush=True)
            modelo = constructor(X_train, algoritmo)
            modelo.fit(X_train, y_train)
            proba = np.asarray(modelo.predict_proba(X_test), dtype=float)
            clases = list(modelo.classes_)
            if (set(clases) != {0, 1} or proba.shape != (len(test), 2)
                    or not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any()
                    or not np.allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
                raise ValueError("predict_proba debe contener P(0), P(1) validas.")
            registrar(nombre, "modelo", proba[:, clases.index(1)])
        mayoria = int(np.bincount(y_train, minlength=2).argmax())
        registrar("Clase_Mayoritaria", "baseline", np.full(len(test), mayoria, dtype=float))
        ocupacion = pd.to_numeric(df.loc[test, "ocupacion_estimada"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(ocupacion).all():
            raise ValueError("Persistencia requiere ocupacion observada valida en t.")
        registrar("Persistencia_Presion", "baseline", (ocupacion > .90).astype(float))
    metricas, por_familia = pd.DataFrame(resultados), pd.DataFrame(estratos)
    resumen = resumir_modelos_binarios(metricas, por_familia)
    metricas.to_csv(destino/"metricas_backtesting_binario.csv", index=False)
    por_familia.to_csv(destino/"metricas_24xx_25xx.csv", index=False)
    resumen.to_csv(destino/"resumen_modelos_binario.csv", index=False)
    metricas[["modelo", "tipo", "anio_prueba", "n_test", "tp", "tn", "fp", "fn"]].to_csv(destino/"matrices_confusion_binario.csv", index=False)
    pd.concat(probas, ignore_index=True).to_csv(destino/"probabilidades_desarrollo_binario.csv", index=False)
    pd.concat(calibracion, ignore_index=True).to_csv(destino/"calibracion_cruda_binario.csv", index=False)
    candidatos = resumen.loc[resumen.tipo.eq("modelo"), "modelo"].tolist()
    (destino/"resumen_metodologico.md").write_text(
        "# Evaluacion binaria temporal de desarrollo\n\n"
        f"Objetivo: `{OBJETIVO}` = 1 exclusivamente si ocupacion observada del siguiente mes exacto > 0.90.\n\n"
        "Interpretacion: condicion futura de alta presion sobre la capacidad de camas.\n\n"
        f"Anios evaluados: {criterio['anios_desarrollo']}. Train utiliza solo periodos objetivo anteriores al anio test. "
        "2025 y 2026 no fueron evaluados ni usados para seleccion.\n\n"
        "Las ventanas vienen de la preparacion mensual previa al recorte t+1. Mediana, moda, escalado y one-hot "
        "se ajustan solo con train. Sin pesos de clase, sample weights, remuestreo ni tuning.\n\n"
        "Umbral 0.50: tecnico provisional para evaluacion, no operacional definitivo. "
        "AP es Average Precision escalonada; ROC requiere ambas clases. Sin positivos AP/recall son NaN; "
        "sin negativos especificidad es NaN; precision/F1 con denominador cero se reportan como 0. "
        "Los baselines usan scores deterministas 0/1; su Brier/AP/ROC no acreditan calibracion probabilistica.\n\n"
        "Calibracion cruda: Brier y 10 intervalos uniformes [0,.1), ..., [.9,1]. No se recalibraron probabilidades.\n\n"
        f"Ranking lexicografico fijado antes del ajuste: {ORDEN_RANKING}; std poblacional ddof=0. "
        "La metrica de familia es el minimo de AP media entre 24xx y 25xx. Baselines separados del ranking.\n\n"
        f"Candidatos en orden: {candidatos}. No constituyen un modelo final de produccion. "
        "No se guardaron estimadores ni se reemplazaron artefactos productivos.\n", encoding="utf-8")
    return metricas, resumen, por_familia, criterio


def cargar_dataset_validado(dataset_path, metadata_path):
    dataset_path, metadata_path = Path(dataset_path), Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("El hash del dataset no coincide con su metadata.")
    politica = metadata.get("tratamiento_capacidad", {})
    if (politica.get("version") != VERSION_POLITICA
            or politica.get("reglas_aplicadas") != ["Q05", "Q06", "Q07", "Q08"]):
        raise ValueError("Dataset sin politica de tratamiento vigente Q05-Q08.")
    target = metadata.get("definicion_target", {})
    if (target.get("version") != VERSION_TARGET or target.get("tipo") != "clasificacion_binaria"
            or target.get("variable_objetivo") != OBJETIVO
            or target.get("indicador") != "ocupacion_siguiente_mes"
            or target.get("formula_ocupacion") != "total_pacientes_camas / total_camas_disponibles"
            or target.get("umbral") != .90 or target.get("comparador") != ">"):
        raise ValueError("Metadata sin definicion binaria vigente estricta >0.90.")
    df = pd.read_csv(dataset_path, float_precision="round_trip", dtype={"codigo_ipress": "string", "id_hospitalizacion": "string", "ubigeo": "string"})
    validar_periodos(df)
    if not df.codigo_ipress.str.fullmatch(r"[0-9]{8}").fillna(False).all():
        raise ValueError("IPRESS debe conservar identidad string canonica de ocho caracteres.")
    validar_target_desarrollo(df)
    return df, {"dataset_path": str(dataset_path.resolve()), "dataset_sha256": _sha256(dataset_path),
                "metadata_path": str(metadata_path.resolve()), "metadata_sha256": _sha256(metadata_path),
                "version_politica": VERSION_POLITICA, "version_target": VERSION_TARGET}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT/"data/processed/dataset_modelo_ipress.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT/"data/processed/dataset_metadata.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solo-plan", action="store_true")
    args = parser.parse_args(argv)
    destino = validar_salida_externa(args.output_dir)
    df, procedencia = cargar_dataset_validado(args.dataset, args.metadata)
    if args.solo_plan:
        destino.mkdir(parents=True, exist_ok=True)
        plan = planificar_folds_binarios(df)
        plan.to_csv(destino/"plan_folds_binario.csv", index=False)
        auditar_features(df).to_csv(destino/"auditoria_features.csv", index=False)
        print(plan.to_string(index=False))
        return 0
    ejecutar_backtesting_binario(df, destino, procedencia=procedencia)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
