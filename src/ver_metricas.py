from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
MATRIZ_PATH = PROJECT_ROOT / "models" / "matriz_confusion_temporal.csv"


def cargar_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cargar_matriz(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}")
    return pd.read_csv(path)


def imprimir_metricas(metadata: dict, matriz: pd.DataFrame) -> None:
    metricas = metadata.get("metricas_temporales", {})
    print("Modelo ganador:", metadata.get("mejor_modelo", "Sin dato"))
    print("Fecha de entrenamiento:", metadata.get("fecha_entrenamiento", "Sin dato"))
    print("Año de prueba:", metadata.get("anio_prueba_temporal", "Sin dato"))
    print(f"Accuracy: {metricas.get('accuracy', 'Sin dato')}")
    print(f"Precision Macro: {metricas.get('precision_macro', 'Sin dato')}")
    print(f"Recall Macro: {metricas.get('recall_macro', 'Sin dato')}")
    print(f"F1 Macro: {metricas.get('f1_macro', 'Sin dato')}")
    print(f"ROC-AUC OVR Macro: {metricas.get('roc_auc_ovr_macro', 'Sin dato')}")
    print("\nMatriz de confusión:")
    print(matriz.to_string(index=False))


def mostrar_heatmap(matriz: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    datos = matriz.set_index(matriz.columns[0])
    sns.heatmap(datos, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Matriz de confusión temporal")
    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()
    plt.show()


def main() -> int:
    metadata = cargar_json(METADATA_PATH)
    matriz = cargar_matriz(MATRIZ_PATH)
    imprimir_metricas(metadata, matriz)
    mostrar_heatmap(matriz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
