import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import entrenar_modelo_binario_final as eb
from src import inferencia_binaria as ib


class PipelineFalso:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        assert list(X.columns) == eb.COLUMNAS_PREDICTORAS_BINARIAS
        assert set(y) == {0, 1}
        return self

    def predict_proba(self, X):
        score = np.where(pd.to_numeric(X["ocupacion_estimada"]) > .8, .8, .2)
        return np.column_stack([1-score, score])


def _dataset():
    filas = []
    for predicho in [*pd.period_range("2025-01", "2025-12", freq="M"), pd.Period("2026-01")]:
        for indice, clase in enumerate((0, 1)):
            fila = {c: float(indice) for c in eb.COLUMNAS_PREDICTORAS_BINARIAS}
            for columna in ("departamento", "provincia", "distrito", "sector", "categoria_ipress"):
                fila[columna] = "categoria"
            fila.update({
                "codigo_ipress": f"{indice+1:08d}",
                "servicio_hospitalizacion": f"servicio-{indice}",
                "periodo_actual": str(predicho-1), "periodo_predicho": str(predicho),
                "familia_servicio": ["hospitalizacion", "cuidados_criticos"][indice],
                "ocupacion_estimada": [.70, .95][indice],
                "ocupacion_siguiente_mes": [.80, 1.00][indice],
                eb.OBJETIVO: clase,
            })
            if predicho.year == 2026:
                fila[eb.OBJETIVO] = "NO_LEER_TARGET_2026"
                fila["ocupacion_siguiente_mes"] = "NO_LEER_OCUPACION_2026"
            filas.append(fila)
    return pd.DataFrame(filas)


def _fuentes(tmp_path):
    root = tmp_path/"repo"
    (root/"data").mkdir(parents=True); (root/"models").mkdir()
    dataset, metadata = root/"data/dataset.csv", root/"data/metadata.json"
    _dataset().to_csv(dataset, index=False)
    metadata.write_text(json.dumps({
        "dataset_sha256": ib.sha256_archivo(dataset),
        "definicion_target": {
            "version": eb.VERSION_TARGET, "variable_objetivo": eb.OBJETIVO,
            "comparador": ">", "umbral": .90,
        },
        "tratamiento_capacidad": {
            "version": eb._metadata_valida.__globals__["VERSION_POLITICA"],
            "reglas_aplicadas": ["Q05", "Q06", "Q07", "Q08"],
        },
    }), encoding="utf-8")
    return root, dataset, metadata


def _evidencia():
    return {
        "desarrollo_2018_2024": {"descripcion": "prueba"},
        "holdout_sellado_2025": {"n": 24, "threshold": .36},
    }


def test_auditoria_final_excluye_target_2026_y_conserva_62_features(tmp_path):
    _, dataset, metadata = _fuentes(tmp_path)
    _, train, y, auditoria = eb.auditar_dataset_final(dataset, metadata)
    assert train.sum() == 24
    assert len(y) == 24 and set(y) == {0, 1}
    assert auditoria["periodo_predicho_maximo"] == "2025-12"
    assert auditoria["registros_excluidos_target_2026"] == 2
    assert auditoria["targets_2026_leidos"] is False
    assert auditoria["numero_features"] == 62
    assert auditoria["lista_features"] == eb.COLUMNAS_PREDICTORAS_BINARIAS


def test_inferencia_valida_features_orden_threshold_y_extras():
    entrada = _dataset().iloc[:2][eb.COLUMNAS_PREDICTORAS_BINARIAS]
    artefacto = {
        "pipeline": PipelineFalso(), "version_modelo": ib.VERSION_MODELO,
        "target": ib.TARGET, "threshold_operacional": .36,
        "features": eb.COLUMNAS_PREDICTORAS_BINARIAS,
    }
    salida = ib.predecir_alta_presion(artefacto, entrada[entrada.columns[::-1]])
    assert salida.probabilidad_alta_presion.between(0, 1).all()
    assert salida.threshold.eq(.36).all()
    assert salida.alerta_alta_presion.equals(salida.probabilidad_alta_presion.ge(.36))
    with pytest.raises(ValueError, match="Faltan"):
        ib.predecir_alta_presion(artefacto, entrada.drop(columns=entrada.columns[0]))
    with pytest.raises(ValueError, match="adicionales"):
        ib.predecir_alta_presion(artefacto, entrada.assign(extra=1))
    permitida = ib.predecir_alta_presion(
        artefacto, entrada.assign(extra=1), permitir_columnas_adicionales=True)
    pd.testing.assert_frame_equal(salida, permitida)


def test_serializacion_metadata_sha_y_proteccion_legacy(tmp_path):
    root, dataset, metadata_dataset = _fuentes(tmp_path)
    legacy_model, legacy_meta = root/"models/modelo_ipress.joblib", root/"models/model_metadata.json"
    legacy_model.write_bytes(b"modelo-legacy"); legacy_meta.write_bytes(b"metadata-legacy")
    antes = legacy_model.read_bytes(), legacy_meta.read_bytes()
    modelo, metadata = root/"models/modelo_ipress_binario_v1.joblib", root/"models/model_metadata_binario_v1.json"
    generado = eb.entrenar_y_serializar(
        dataset, metadata_dataset, tmp_path/"holdout", modelo, metadata,
        algoritmo=object(), constructor_pipeline=lambda X, algoritmo: PipelineFalso(),
        evidencia=_evidencia(), root=root,
    )
    assert antes == (legacy_model.read_bytes(), legacy_meta.read_bytes())
    assert generado["numero_observaciones"] == 24
    assert generado["threshold_operacional"] == .36
    assert generado["modelo_sha256"] == ib.sha256_archivo(modelo)
    assert generado["smoke_test_serializacion"]["probabilidades_coinciden"] is True
    assert generado["smoke_test_serializacion"]["alertas_coinciden"] is True
    artefacto, metadata_cargada = ib.cargar_modelo_binario(modelo, metadata)
    assert metadata_cargada["lista_features"] == eb.COLUMNAS_PREDICTORAS_BINARIAS
    assert joblib.load(modelo)["threshold_operacional"] == .36
    entrada = _dataset().iloc[:2][eb.COLUMNAS_PREDICTORAS_BINARIAS]
    assert ib.predecir_alta_presion(artefacto, entrada).threshold.eq(.36).all()
    with pytest.raises(FileExistsError):
        eb.entrenar_y_serializar(
            dataset, metadata_dataset, tmp_path/"holdout", modelo, metadata,
            evidencia=_evidencia(), root=root)


def test_prohibe_rutas_legacy(tmp_path):
    root, dataset, metadata = _fuentes(tmp_path)
    with pytest.raises(ValueError, match="legacy"):
        eb.entrenar_y_serializar(
            dataset, metadata, tmp_path/"holdout",
            root/"models/modelo_ipress.joblib", root/"models/otro.json",
            evidencia=_evidencia(), root=root)
