import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import evaluar_reglas_decision as rd
from test_reglas_decision import datos, motor_falso


def preparar(tmp_path):
    df = datos()
    destino = tmp_path/'models'
    destino.mkdir()
    ruta = tmp_path/'data/processed/dataset_modelo_ipress.csv'
    ruta.parent.mkdir(parents=True)
    df.to_csv(ruta, index=False)
    huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
    (ruta.parent/'dataset_metadata.json').write_text(json.dumps({'dataset_sha256': huella}))
    historico, folds, _ = rd.ef.preparar_desarrollo(df)
    final = df.loc[df.periodo_predicho.lt('2026-01')].copy()
    f2025 = next(f for f in rd.bt.crear_folds_expansivos(final)[0] if f[0] == 2025)
    matrices, filas = {}, []
    for d, fold, fase in [(historico, f, 'desarrollo') for f in folds]+[(final, f2025, 'comprobacion_2025')]:
        anio, train, test = fold
        clases = d.loc[test, rd.bt.OBJETIVO].to_numpy()
        p = np.array([[.7, .1, .2], [.2, .5, .3], [.4, .25, .35]])[clases]
        matrices[f'p_{anio}'] = p
        filas.append(rd.evaluar_probabilidades(d, train, test, p, anio,
            [rd.reglas_candidatas()[0]], fase))
    pd.concat(filas, ignore_index=True).to_csv(destino/rd.ARCHIVOS[0], index=False)
    motor = motor_falso()
    (destino/rd.ARCHIVOS[2]).write_text(json.dumps({'procedencia': {'dataset_sha256': huella},
        'anios_desarrollo': list(rd.ANIOS_DESARROLLO), 'conjunto_features': 'D',
        'columnas_predictoras': [*motor.COLUMNAS_PREDICTORAS, *rd.CONJUNTOS['D']],
        'hiperparametros_sin_modificar': motor.obtener_modelos()['XGBoost'].get_params()}))
    for n in [rd.ARCHIVOS[1], rd.ARCHIVOS[3], 'modelo_ipress.joblib']:
        (destino/n).write_bytes(b'original protegido')
    return df, destino, huella, matrices, motor


@pytest.mark.parametrize('persistidas', [False, True])
def test_cli_extension_aislada_congelada_y_origen_verificable(tmp_path, monkeypatch, persistidas):
    df, destino, _, matrices, motor = preparar(tmp_path)
    antes = {p: p.read_bytes() for p in destino.iterdir()}
    if persistidas:
        np.savez_compressed(destino/'probabilidades_reglas_decision.npz', **matrices)
    monkeypatch.setattr(rd.bt, 'ROOT', tmp_path)
    def obtener_motor():
        if persistidas:
            raise AssertionError('No ajustar cuando se reutilizan matrices.')
        return motor
    monkeypatch.setattr(rd.bt, '_motor_existente', obtener_motor)
    original = rd.probabilidades_fold
    def observar(motor, algoritmo, d, train, test, columnas, anio):
        if anio == 2025:
            j = json.loads((destino/rd.ARCHIVOS_EXTENSION[2]).read_text())
            assert j['regla_seleccionada'] is not None
            assert j['evaluacion_2025'] is None
            assert pd.read_csv(destino/rd.ARCHIVOS_EXTENSION[0]).anio_prueba.max() == 2024
        return original(motor, algoritmo, d, train, test, columnas, anio)
    monkeypatch.setattr(rd, 'probabilidades_fold', observar)
    monkeypatch.setattr(sys, 'argv', ['evaluar_reglas_decision', '--extension-020'])
    rd.main()
    assert antes == {p: p.read_bytes() for p in antes}
    r = pd.read_csv(destino/rd.ARCHIVOS_EXTENSION[0])
    s = pd.read_csv(destino/rd.ARCHIVOS_EXTENSION[1])
    j = json.loads((destino/rd.ARCHIVOS_EXTENSION[2]).read_text())
    assert len(s) == 6 and 'reduce_ambos_errores' in s
    assert len(r.loc[r.fase.eq('desarrollo')]) == 30
    assert set(r.regla) == {r['regla'] for r in rd.reglas_extension_020()}
    assert j['2025_participo_en_seleccion'] is False
    assert j['es_modelo_final_produccion'] is False
    assert len(j['verificacion_por_fold']) == 6
    origen = 'reutilizadas' if persistidas else 'reproducidas'
    assert {f['origen_probabilidades'] for f in j['verificacion_por_fold']} == {origen}
    assert all(f['argmax_reproducido'] for f in j['verificacion_por_fold'])
    assert len([e for e, _, _ in motor.eventos if e == 'fit']) == (0 if persistidas else 6)
    assert len([e for e, _, _ in motor.eventos if e == 'proba']) == (0 if persistidas else 6)
    with np.load(destino/rd.ARCHIVOS_EXTENSION[3], allow_pickle=False) as guardadas:
        assert set(guardadas.files) == set(matrices)
    with pytest.raises(FileExistsError):
        rd.main()


@pytest.mark.parametrize('extension', [False, True])
def test_cli_solo_plan_elige_grilla_sin_ajustar_ni_escribir(tmp_path, monkeypatch, capsys, extension):
    _, destino, _, _, _ = preparar(tmp_path)
    antes = set(destino.iterdir())
    monkeypatch.setattr(rd.bt, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['m', '--solo-plan']+(['--extension-020'] if extension else []))
    rd.main()
    salida = json.loads(capsys.readouterr().out)
    assert len(salida['reglas']) == (6 if extension else 12)
    assert set(destino.iterdir()) == antes


def test_cli_sin_flag_conserva_ruta_original(tmp_path, monkeypatch):
    preparar(tmp_path)
    monkeypatch.setattr(rd.bt, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['m'])
    class Original(Exception):
        pass
    def original(*a, **k):
        raise Original()
    monkeypatch.setattr(rd, 'evaluar_reglas', original)
    with pytest.raises(Original):
        rd.main()


@pytest.mark.parametrize('problema', ['metricas', 'test', 'parametros', 'matriz'])
def test_divergencias_abortan_sin_procesar_2025(tmp_path, problema):
    df, destino, huella, matrices, motor = preparar(tmp_path)
    if problema in ['metricas', 'test']:
        r = pd.read_csv(destino/rd.ARCHIVOS[0])
        r.loc[0, 'f1_macro' if problema == 'metricas' else 'test_sha256'] = .123 if problema == 'metricas' else 'distinto'
        r.to_csv(destino/rd.ARCHIVOS[0], index=False)
    elif problema == 'parametros':
        path = destino/rd.ARCHIVOS[2]
        j = json.loads(path.read_text())
        j['hiperparametros_sin_modificar']['random_state'] = 99
        path.write_text(json.dumps(j))
    else:
        matrices['p_2018'] = matrices['p_2018'][:, [2, 1, 0]]
        np.savez_compressed(destino/'probabilidades_reglas_decision.npz', **matrices)
    with pytest.raises(ValueError):
        rd.ejecutar_extension_020(df, destino, dataset_sha256=huella, motor=motor)
    assert len([e for e, _, _ in motor.eventos if e == 'fit']) <= 1
    if (destino/rd.ARCHIVOS_EXTENSION[2]).exists():
        j = json.loads((destino/rd.ARCHIVOS_EXTENSION[2]).read_text())
        assert j['evaluacion_2025'] is None and j['regla_seleccionada'] is None


def test_metricas_iguales_no_se_declaran_probabilidades_reutilizadas(tmp_path, monkeypatch):
    df, destino, huella, _, motor = preparar(tmp_path)
    funcion = rd.probabilidades_fold
    def otra_matriz(*a):
        p = funcion(*a).copy()
        # Diferencia ínfima que no altera decisiones/AUC; cambia identidad binaria.
        p[:, 0] += 1e-14
        p[:, 1] -= 1e-14
        return p
    monkeypatch.setattr(rd, 'probabilidades_fold', otra_matriz)
    _, _, j = rd.ejecutar_extension_020(df, destino, dataset_sha256=huella, motor=motor)
    assert all(f['origen_probabilidades'] == 'reproducidas' for f in j['verificacion_por_fold'])
    assert not any(f['matriz_identica_por_hash'] for f in j['verificacion_por_fold'])


def test_flag_archivo_requiere_extension(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['m', '--probabilidades-originales', 'x.npz'])
    with pytest.raises(SystemExit) as e:
        rd.main()
    assert e.value.code == 2
