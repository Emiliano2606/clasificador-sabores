"""Utilidades de limpieza y preparación del dataset de sabores."""

from __future__ import annotations

import pandas as pd


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset: normaliza nombres de columnas, elimina duplicados
    y descarta filas con valores nulos en columnas críticas.

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame limpio.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates().dropna()
    return df.reset_index(drop=True)


def separar_xy(df: pd.DataFrame, objetivo: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separa características (X) y variable objetivo (y)."""
    if objetivo not in df.columns:
        raise KeyError(f"La columna objetivo '{objetivo}' no existe en el dataset.")
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    return X, y
