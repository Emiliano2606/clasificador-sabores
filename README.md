# Clasificador de Sabores

Clasificación de sabores mediante aprendizaje supervisado, comparando **SVM** y **Random Forest**.

## Estructura

```
clasificador-sabores/
├── notebooks/    # Flujo de trabajo paso a paso
├── src/          # Código reutilizable (preprocesamiento, features, predicción)
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+
- Dependencias: `pip install -r requirements.txt`

## Flujo de trabajo

1. `01_revision_dataset.ipynb` — inspección y validación de datos.
2. `02_extraccion_caracteristicas.ipynb` — limpieza y matriz de características.
3. `03_modelo_svm.ipynb` — entrenamiento y ajuste del SVM.
4. `04_modelo_random_forest.ipynb` — entrenamiento y ajuste del Random Forest.
5. `05_evaluacion.ipynb` — comparación y selección del modelo final.

## Uso

```bash
python src/predict.py --modelo modelos/modelo.joblib --datos datos/nuevos.csv
```

## Licencia

MIT
