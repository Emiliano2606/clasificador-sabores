
from pathlib import Path

import joblib
import numpy as np

from features import extract_features


# Rutas del proyecto
CARPETA_SRC = Path(__file__).resolve().parent
CARPETA_PROYECTO = CARPETA_SRC.parent

RUTA_MODELO = (
    CARPETA_PROYECTO
    / "models"
    / "svm_multietiqueta.joblib"
)

# Etiquetas utilizadas durante el entrenamiento
SABORES = [
    "dulce",
    "salado",
    "amargo",
    "acido",
    "umami"
]


def cargar_modelo():
    """Carga el modelo SVM entrenado."""

    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {RUTA_MODELO}"
        )

    return joblib.load(RUTA_MODELO)


def extraer_caracteristicas(ruta_imagen):
    """Valida la imagen y extrae sus características."""

    ruta_imagen = Path(ruta_imagen)

    if not ruta_imagen.is_file():
        raise FileNotFoundError(
            f"No se encontró la imagen: {ruta_imagen}"
        )

    caracteristicas = extract_features(str(ruta_imagen))

    X = np.asarray(
        caracteristicas,
        dtype=np.float32
    ).reshape(1, -1)

    if not np.isfinite(X).all():
        raise ValueError(
            "Las características contienen valores no válidos."
        )

    return X


def analizar_prediccion(ruta_imagen):
    """Predice los sabores y muestra las puntuaciones del SVM."""

    X = extraer_caracteristicas(ruta_imagen)
    modelo = cargar_modelo()

    # Obtener las predicciones
    predicciones = modelo.predict(X)[0]

    # Obtener las puntuaciones de decisión
    if not hasattr(modelo, "decision_function"):
        raise AttributeError(
            "El modelo no permite obtener puntuaciones de decisión."
        )

    puntuaciones = np.asarray(
        modelo.decision_function(X)
    ).reshape(-1)

    if len(predicciones) != len(SABORES):
        raise ValueError(
            "La cantidad de predicciones no coincide con los sabores."
        )

    if len(puntuaciones) != len(SABORES):
        raise ValueError(
            "La cantidad de puntuaciones no coincide con los sabores."
        )

    sabores_detectados = [
        sabor
        for sabor, valor in zip(SABORES, predicciones)
        if valor == 1
    ]

    print("\n" + "=" * 48)
    print("RESULTADO DE LA CLASIFICACIÓN")
    print("=" * 48)

    if sabores_detectados:
        print(
            "Sabores predichos:",
            ", ".join(sabores_detectados)
        )
    else:
        print("El modelo no predijo ninguno de los cinco sabores.")

    print("\nPuntuaciones por sabor:")
    print("-" * 48)

    resultados = sorted(
        zip(SABORES, puntuaciones, predicciones),
        key=lambda elemento: elemento[1],
        reverse=True
    )

    for sabor, puntuacion, prediccion in resultados:
        estado = (
            "Predicho"
            if prediccion == 1
            else "No predicho"
        )

        print(
            f"{sabor:8} | "
            f"Puntuación: {puntuacion:8.3f} | "
            f"{estado}"
        )

    print("-" * 48)
    print(
        "Nota: las puntuaciones del SVM no son "
        "probabilidades ni porcentajes."
    )

    return sabores_detectados


if __name__ == "__main__":
    print("CLASIFICADOR DE SABORES")
    print("-" * 48)

    ruta = input(
        "Escribe la ruta de una imagen: "
    ).strip().strip('"')

    try:
        analizar_prediccion(ruta)

    except (FileNotFoundError, ValueError, AttributeError) as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(
            f"\nOcurrió un error inesperado: "
            f"{type(error).__name__}: {error}"
        )
#:)  
