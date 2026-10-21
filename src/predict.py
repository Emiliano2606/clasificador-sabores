"""Inferencia: carga un modelo entrenado y predice sobre nuevos datos."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def cargar_modelo(ruta: str | Path):
    """Carga un modelo serializado con joblib."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {ruta}")
    return joblib.load(ruta)


def predecir(modelo, datos: pd.DataFrame):
    """Devuelve las predicciones y, si el modelo lo permite, las probabilidades."""
    predicciones = modelo.predict(datos)
    probabilidades = None
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(datos)
    return predicciones, probabilidades


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasificador de sabores")
    parser.add_argument("--modelo", required=True, help="Ruta al modelo .joblib")
    parser.add_argument("--datos", required=True, help="Ruta a un CSV con datos nuevos")
    args = parser.parse_args()

    modelo = cargar_modelo(args.modelo)
    datos = pd.read_csv(args.datos)
    predicciones, _ = predecir(modelo, datos)

    print(f"Predicciones generadas: {len(predicciones)}")
    print(predicciones[:10])


if __name__ == "__main__":
    main()
