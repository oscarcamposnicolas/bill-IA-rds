"""
Módulo de Entrenamiento del Clasificador de Contexto (Fase 4, Paso 2).

Este script entrena el modelo de Machine Learning (ML) Random Forest para la tarea
de Clasificación de Contexto ('Clásico' vs 'Black Edition').

El clasificador utiliza el Meta-Dataset generado por el modelo YOLO, basando su
decisión únicamente en el conteo de bolas de cada tipo detectado en la mesa.

Propósito principal:
1.  Construir y persistir el clasificador final de la Fase 4.
2.  Evaluar el rendimiento del ML clásico sobre las características extraídas por el DL.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURACIÓN ---
META_DATASET_PATH = "RandomForestClassifier/meta_dataset.csv"
MODELO_SALIDA_PATH = "RandomForestClassifier/context_classifier.joblib"
# --- FIN DE LA CONFIGURACIÓN ---


def entrenar_meta_modelo():
    # Cargar el dataset que hemos creado
    try:
        df = pd.read_csv(META_DATASET_PATH)
        print(
            f"Dataset cargado exitosamente desde '{META_DATASET_PATH}'. Contiene {len(df)} filas."
        )
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{META_DATASET_PATH}'.")
        return

    # Separar las características (X) del objetivo (y)
    # Usamos todas las columnas como características excepto el path de la imagen y el target
    X = df.drop(columns=["image_path", "target"])
    y = df["target"]

    # Dividir los datos en conjuntos de entrenamiento y prueba para evaluar nuestro clasificador
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(
        f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba."
    )

    # Crear y entrenar el modelo Random Forest
    # class_weight='balanced' es la clave para manejar el desequilibrio de clases
    print("\nIniciando entrenamiento del clasificador de contexto (Random Forest)...")
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("¡Entrenamiento completado!")

    # Evaluar el modelo
    print("\n--- Evaluación del Clasificador de Contexto ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión (Accuracy): {accuracy:.4f}")

    print("\nInforme de Clasificación:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Classic (0)", "Black Edition (1)"]
        )
    )

    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar el modelo entrenado para usarlo en la inferencia final
    joblib.dump(model, MODELO_SALIDA_PATH)
    print(f"\n¡Modelo clasificador de contexto guardado en '{MODELO_SALIDA_PATH}'!")


if __name__ == "__main__":
    entrenar_meta_modelo()
