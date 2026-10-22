
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


# Configuración general
IMAGE_SIZE = (64, 64)

# Configuración de HOG
HOG_ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)

# Configuración de LBP
LBP_POINTS = 24
LBP_RADIUS = 3
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2


def load_image(image_path):
    """
    Carga una imagen, la redimensiona y genera
    sus versiones de color y escala de grises.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"No se pudo leer la imagen: {image_path}"
        )

    # OpenCV carga en BGR; convertimos a RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Todas las imágenes tendrán las mismas dimensiones.
    image = cv2.resize(image, IMAGE_SIZE)

    # Conversión a escala de grises.
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    return image, gray


def extract_color_histogram(image):
    """
    Calcula histogramas normalizados en el espacio HSV.
    Utiliza 8 intervalos por canal: 24 características.
    """
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    histograms = []

    for channel, bins, limits in [
        (0, 8, [0, 180]),
        (1, 8, [0, 256]),
        (2, 8, [0, 256])
    ]:
        histogram = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [bins],
            limits
        )

        histogram = cv2.normalize(
            histogram,
            histogram
        )

        histograms.extend(
            histogram.flatten()
        )

    return np.asarray(
        histograms,
        dtype=np.float32
    )


def extract_hog(gray):
    """
    Extrae características HOG para representar
    bordes, gradientes y estructura visual.
    """
    gray_normalized = gray.astype(
        np.float32
    ) / 255.0

    features = hog(
        gray_normalized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True
    )

    return features.astype(np.float32)


def extract_lbp(gray):
    """
    Extrae un histograma LBP para describir
    patrones locales de textura.
    """
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(
            0,
            LBP_BINS + 1
        ),
        range=(0, LBP_BINS),
        density=True
    )

    return histogram.astype(np.float32)


def extract_features(image_path):
    """
    Concatena color, HOG y LBP en un solo vector.
    """
    image, gray = load_image(image_path)

    color_features = extract_color_histogram(image)
    hog_features = extract_hog(gray)
    lbp_features = extract_lbp(gray)

    features = np.concatenate([
        color_features,
        hog_features,
        lbp_features
    ])

    return features.astype(np.float32)


def extract_dataset(image_paths):
    """
    Procesa una colección de imágenes.

    Devuelve:
    X: matriz de características.
    valid_paths: rutas de las imágenes procesadas.
    errors: imágenes que no pudieron procesarse.
    """
    feature_vectors = []
    valid_paths = []
    errors = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            vector = extract_features(image_path)

            feature_vectors.append(vector)
            valid_paths.append(str(image_path))

        except Exception as error:
            errors.append({
                "ruta": str(image_path),
                "error": str(error)
            })

        if index % 100 == 0:
            print(
                f"Procesadas {index} de "
                f"{len(image_paths)} imágenes"
            )

    if not feature_vectors:
        raise ValueError(
            "No se pudo extraer ninguna imagen."
        )

    X = np.vstack(feature_vectors)

    return X, valid_paths, errors
