"""Backtesting expansivo de evaluación; nunca guarda un modelo de producción."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OBJETIVO = "nivel_riesgo_siguiente_mes_codificado"
ETIQUETAS = {"nivel_riesgo_actual", "nivel_riesgo_actual_codificado",
             "nivel_riesgo_siguiente_mes", OBJETIVO, "nivel_riesgo", "nivel_riesgo_codificado"}
METRICAS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
            "specificity_macro", "roc_auc_ovr_macro", "recall_alto", "precision_alto", "f1_alto",
            "casos_alto_reales", "falsos_negativos_alto", "tasa_falsos_negativos_alto",
            "errores_alto_bajo", "proporcion_alto_bajo", "proporcion_alto_bajo_total"]
ORDEN_COMPARACION = ["f1_macro_promedio", "balanced_accuracy_promedio",
                     "recall_alto_promedio", "tasa_falsos_negativos_alto_promedio"]


def _clases(valores):
    datos = np.asarray(valores)
    if datos.ndim != 1 or not np.isin(datos, [0, 1, 2]).all():
        raise ValueError("Se requieren etiquetas 0, 1 o 2, sin valores ausentes.")
    return datos.astype(int)


def validar_periodos(df):
    if df.empty or not df.index.is_unique:
        raise ValueError("Dataset vacío o índices duplicados.")
    actual = pd.PeriodIndex(df["periodo_actual"].astype(str), freq="M")
    objetivo = pd.PeriodIndex(df["periodo_predicho"].astype(str), freq="M")
    if actual.hasnans or objetivo.hasnans or not (objetivo == actual + 1).all():
        raise ValueError("Cada objetivo debe corresponder exactamente a t+1.")
    if df.duplicated(["codigo_ipress", "servicio_hospitalizacion", "periodo_actual"]).any():
        raise ValueError("Hay periodos duplicados por IPRESS y servicio.")
    _clases(df[OBJETIVO])
    return actual, objetivo


def planificar_folds(df, min_meses_historial=24, min_casos_clase=2):
    """Año completo = 12 meses objetivo presentes en el conjunto, no por IPRESS.

    El historial cuenta periodos objetivo distintos anteriores al año de test;
    los huecos globales no invalidan un fold con suficiente historial.
    """
    if min_meses_historial < 1 or min_casos_clase < 1:
        raise ValueError("Los mínimos deben ser positivos.")
    _, periodos = validar_periodos(df)
    filas = []
    for anio in sorted(set(periodos.year)):
        train, test = periodos.year < anio, periodos.year == anio
        meses_train = set(periodos[train])
        conteos = df.loc[train, OBJETIVO].value_counts()
        motivos = []
        if set(periodos[test].month) != set(range(1, 13)):
            motivos.append("Año objetivo incompleto")
        if len(meses_train) < min_meses_historial:
            motivos.append(f"Se requieren al menos {min_meses_historial} periodos mensuales distintos anteriores")
        if any(conteos.get(c, 0) < min_casos_clase for c in (0, 1, 2)):
            motivos.append(f"Se requieren al menos {min_casos_clase} casos de cada clase en train")
        filas.append({"anio_prueba": int(anio), "elegible": not motivos,
                      "motivo": "; ".join(motivos) or "Cumple criterios",
                      "n_train": int(train.sum()), "n_test": int(test.sum()),
                      "meses_train": len(meses_train), "meses_test": len(set(periodos[test])),
                      "train_desde": str(periodos[train].min()) if train.any() else "",
                      "train_hasta": str(periodos[train].max()) if train.any() else "",
                      **{f"train_clase_{c}": int(conteos.get(c, 0)) for c in (0, 1, 2)}})
    return pd.DataFrame(filas)


def crear_folds_expansivos(df, min_meses_historial=24, min_casos_clase=2):
    plan = planificar_folds(df, min_meses_historial, min_casos_clase)
    _, periodos = validar_periodos(df)
    folds = [(int(anio), df.index[periodos.year < anio], df.index[periodos.year == anio])
             for anio in plan.loc[plan.elegible, "anio_prueba"]]
    return folds, plan


def calcular_metricas_backtesting(y_real, y_pred, probabilidades=None):
    """Macros sobre las tres clases. AUC OVR por rangos con empates promediados.

    Sin casos Alto, recall/F1/FNR Alto y proporción Alto->Bajo son NaN.
    Precision Alto = 0 si no hay predicciones Alto. AUC vacío sin probabilidades
    o sin las tres clases reales; no inventa probabilidades para baselines.
    """
    real, pred = _clases(y_real), _clases(y_pred)
    if len(real) == 0 or len(real) != len(pred):
        raise ValueError("Las predicciones deben cubrir exactamente el test.")
    matriz = np.zeros((3, 3), dtype=int)
    np.add.at(matriz, (real, pred), 1)
    soporte, estimados, tp = matriz.sum(axis=1), matriz.sum(axis=0), matriz.diagonal()
    recall = np.divide(tp, soporte, out=np.zeros(3), where=soporte != 0)
    precision = np.divide(tp, estimados, out=np.zeros(3), where=estimados != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=(precision+recall) != 0)
    tn = len(real)-soporte-estimados+tp
    especificidad = np.divide(tn, len(real)-soporte, out=np.zeros(3), where=(len(real)-soporte) != 0)
    auc = np.nan
    if probabilidades is not None:
        proba = np.asarray(probabilidades, dtype=float)
        if (proba.shape != (len(real), 3) or not np.isfinite(proba).all()
                or (proba < 0).any() or (proba > 1).any()
                or not np.allclose(proba.sum(axis=1), 1)):
            raise ValueError("Probabilidades inválidas; columnas esperadas: 0, 1, 2.")
        if (soporte > 0).all():
            aucs = []
            for c in range(3):
                rangos = pd.Series(proba[:, c]).rank(method="average").to_numpy()
                positivos = soporte[c]
                aucs.append((rangos[real == c].sum()-positivos*(positivos+1)/2)
                            / (positivos*(len(real)-positivos)))
            auc = float(np.mean(aucs))
    altos, fn, severos = int(soporte[2]), int(soporte[2]-tp[2]), int(matriz[2, 0])
    return {"accuracy": float(tp.sum()/len(real)),
            "balanced_accuracy": float(recall[soporte > 0].mean()),
            "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()), "specificity_macro": float(especificidad.mean()),
            "roc_auc_ovr_macro": auc, "recall_alto": float(recall[2]) if altos else np.nan,
            "precision_alto": float(precision[2]), "f1_alto": float(f1[2]) if altos else np.nan,
            "casos_alto_reales": altos, "falsos_negativos_alto": fn,
            "tasa_falsos_negativos_alto": fn/altos if altos else np.nan,
            "errores_alto_bajo": severos, "proporcion_alto_bajo": severos/altos if altos else np.nan,
            "proporcion_alto_bajo_total": severos/len(real)}


def predicciones_baselines(df_test, y_train):
    mayoria = int(pd.Series(y_train).value_counts().sort_index().idxmax())
    return {"Clase_Mayoritaria": np.full(len(df_test), mayoria),
            "Persistencia_Riesgo_Actual": _clases(df_test["nivel_riesgo_actual_codificado"]),
            "Regla_Ocupacion_Actual": np.select([df_test.ocupacion_estimada.ge(0.85),
                                                  df_test.ocupacion_estimada.ge(0.70)], [2, 1], default=0)}


def resumir_backtesting(metricas):
    if metricas.empty:
        raise ValueError("No hay resultados para resumir.")
    if metricas.duplicated(["modelo", "anio_prueba"]).any():
        raise ValueError("Resultados duplicados por modelo y año.")
    conjuntos = metricas.groupby("modelo").anio_prueba.agg(lambda x: frozenset(x))
    if len(set(conjuntos)) != 1:
        raise ValueError("No se comparan modelos evaluados en años distintos.")
    for _, grupo in metricas.groupby("anio_prueba"):
        if grupo.n_test.nunique() != 1 or grupo.test_sha256.nunique() != 1:
            raise ValueError("Los modelos deben usar exactamente el mismo test.")
    filas = []
    for nombre, grupo in metricas.groupby("modelo", sort=True):
        fila = {"modelo": nombre, "tipo": grupo.tipo.iloc[0], "n_anios": len(grupo)}
        for metrica in METRICAS:
            valores = grupo[metrica]
            fila.update({f"{metrica}_promedio": valores.mean(), f"{metrica}_std": valores.std(ddof=0),
                         f"{metrica}_min": valores.min(), f"{metrica}_max": valores.max(),
                         f"{metrica}_n_validos": int(valores.notna().sum())})
        filas.append(fila)
    return pd.DataFrame(filas).sort_values(ORDEN_COMPARACION, ascending=[False, False, False, True],
                                          na_position="last", kind="stable").reset_index(drop=True)


def mejores_comparados(resumen):
    """Desempate lexicográfico por métricas, nunca por nombre de algoritmo."""
    mejor = resumen.iloc[0]
    empate = pd.Series(True, index=resumen.index)
    for columna in ORDEN_COMPARACION:
        empate &= resumen[columna].eq(mejor[columna]) | (resumen[columna].isna() & pd.isna(mejor[columna]))
    return resumen.loc[empate, "modelo"].tolist()


def _motor_existente():
    # Importación diferida: planificar y comprobar métricas no requieren sklearn.
    if __package__:
        from . import entrenar_modelo
    else:
        import entrenar_modelo
    return entrenar_modelo


def ejecutar_backtesting(df, output_dir=None, *, min_meses_historial=24, min_casos_clase=2, motor=None):
    """Ajusta modelos de evaluación por fold; no llama a entrenar_modelos/guardar_resultados."""
    folds, plan = crear_folds_expansivos(df, min_meses_historial, min_casos_clase)
    if not folds:
        raise ValueError("Ningún año elegible; consulte planificar_folds.")
    motor = motor if motor is not None else _motor_existente()
    columnas = list(motor.COLUMNAS_PREDICTORAS)
    if ETIQUETAS.intersection(columnas) or set(motor.COLUMNAS_EXCLUIDAS).intersection(columnas):
        raise ValueError("Columnas objetivo o excluidas presentes entre predictores.")
    algoritmos = motor.obtener_modelos()
    if not algoritmos:
        raise ValueError("No hay algoritmos disponibles.")
    filas = []
    for anio, train, test in folds:
        X_train, X_test = df.loc[train, columnas].copy(), df.loc[test, columnas].copy()
        y_train, y_test = df.loc[train, OBJETIVO], df.loc[test, OBJETIVO]
        huella = hashlib.sha256(pd.util.hash_pandas_object(
            df.loc[test, ["codigo_ipress", "servicio_hospitalizacion", "periodo_predicho", OBJETIVO]],
            index=True).to_numpy().tobytes()).hexdigest()
        def registrar(nombre, tipo, pred, proba=None):
            filas.append({"modelo": nombre, "tipo": tipo, "anio_prueba": anio,
                          "n_train": len(train), "n_test": len(test), "test_sha256": huella,
                          **calcular_metricas_backtesting(y_test, pred, proba)})
        for nombre, algoritmo in algoritmos.items():
            print(f"Backtesting {anio}: {nombre}", flush=True)
            modelo = motor.crear_pipeline(X_train, algoritmo)
            motor.ajustar_pipeline(nombre, modelo, X_train, y_train)
            pred = modelo.predict(X_test)
            proba = None
            if hasattr(modelo, "predict_proba"):
                proba = np.asarray(modelo.predict_proba(X_test))
                clases = list(modelo.classes_)
                proba = proba[:, [clases.index(c) for c in (0, 1, 2)]]
            registrar(nombre, "modelo", pred, proba)
        for nombre, pred in predicciones_baselines(df.loc[test], y_train).items():
            registrar(nombre, "baseline", pred)
    metricas = pd.DataFrame(filas)
    resumen = resumir_backtesting(metricas)
    comparacion = {"orden_criterios": ORDEN_COMPARACION,
                  "sentido": ["max", "max", "max", "min"],
                  "mejores_incluidos_baselines": mejores_comparados(resumen),
                  "mejores_modelos": mejores_comparados(resumen.loc[resumen.tipo == "modelo"]),
                  "anios_elegibles": [f[0] for f in folds],
                  "min_meses_historial": min_meses_historial, "min_casos_clase": min_casos_clase,
                  "xgboost_disponible": "XGBoost" in algoritmos,
                  "modelo_produccion_modificado": False}
    if output_dir is not None:
        destino = Path(output_dir)
        destino.mkdir(parents=True, exist_ok=True)
        metricas.to_csv(destino/"metricas_backtesting_temporal.csv", index=False)
        resumen.to_csv(destino/"resumen_backtesting_temporal.csv", index=False)
        plan.to_csv(destino/"plan_backtesting_temporal.csv", index=False)
        (destino/"comparacion_backtesting_temporal.json").write_text(
            json.dumps(comparacion, ensure_ascii=False, indent=2), encoding="utf-8")
    return metricas, resumen, comparacion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-plan", action="store_true", help="No ajusta modelos; documenta años elegibles.")
    parser.add_argument("--min-meses-historial", type=int, default=24)
    parser.add_argument("--min-casos-clase", type=int, default=2)
    args = parser.parse_args()
    archivo = ROOT/"data/processed/dataset_modelo_ipress.csv"
    contenido = archivo.read_bytes()
    huella = hashlib.sha256(contenido).hexdigest()
    metadata = json.loads((ROOT/"data/processed/dataset_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != huella:
        raise ValueError("Dataset y metadata no coinciden; no se ejecuta backtesting.")
    import io
    df = pd.read_csv(io.BytesIO(contenido), dtype={"codigo_ipress": str, "servicio_hospitalizacion": str})
    plan = planificar_folds(df, args.min_meses_historial, args.min_casos_clase)
    print(plan.to_string(index=False))
    if args.solo_plan:
        (ROOT/"models").mkdir(exist_ok=True)
        plan.to_csv(ROOT/"models/plan_backtesting_temporal.csv", index=False)
    else:
        _, _, comparacion = ejecutar_backtesting(df, ROOT/"models",
            min_meses_historial=args.min_meses_historial, min_casos_clase=args.min_casos_clase)
        comparacion["dataset_sha256"] = huella
        comparacion["definicion_target"] = metadata.get("definicion_target", {})
        (ROOT/"models/comparacion_backtesting_temporal.json").write_text(
            json.dumps(comparacion, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(comparacion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
