"""Extracción y transformación de características para el clasificador."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def construir_caracteristicas(
    df: pd.DataFrame,
    objetivo: str,
) -> Pipeline:
    """Construye un pipeline reproducible de preprocesamiento.

    Escala variables numéricas y codifica variables categóricas con One-Hot.

    Args:
        df: DataFrame limpio.
        objetivo: Nombre de la columna objetivo.

    Returns:
        Pipeline de scikit-learn sin entrenar, con los pasos de transformación.
    """
    columnas = [c for c in df.columns if c != objetivo]
    numericas = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
    categoricas = [c for c in columnas if c not in numericas]

    transformador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocesamiento", transformador)])
