"""
Módulo de Benchmarking y Test de Lote (Fase 2, Paso 3).

Este script automatiza la inferencia del modelo YOLO de Detección de Bolas sobre
una carpeta completa de imágenes de prueba. Es fundamental para:

1.  Benchmarking: Medir la confianza promedio y la latencia del modelo.
2.  Validación Visual: Generar imágenes de resultados con bounding boxes para
    la documentación y el informe final.
3.  Prueba de Robustez: Evaluar el rendimiento del modelo en diferentes escenarios
    de iluminación o tipos de mesa (Clásico/Black Edition).
"""

import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN ---

# --- Rutas del Proyecto ---
base_project_dir = os.path.join(".", "detect_balls")
train_version = "Supermodelo_Fase2_Final"
train_version = "Modelo_Hibrido_v1"

runs_dir = os.path.join(base_project_dir, "runs")

# --- Carpetas con las imágenes de prueba ---
TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Classic")
TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Black")
TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Mix")
TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Classic2")

# Carpeta donde se guardarán los resultados de este script
output_dir = os.path.join(base_project_dir, "tests", "results_classic")
output_dir = os.path.join(base_project_dir, "tests", "results_black")
output_dir = os.path.join(base_project_dir, "tests", "results_mix")
output_dir = os.path.join(base_project_dir, "tests", "results_classic2")

# --- Ruta al Modelo ---
# Apunta al mejor modelo del entrenamiento final
model_path = os.path.join(
    runs_dir,
    train_version,
    "weights",
    "best.pt",
)

# --- Parámetros de Inferencia ---
CONFIDENCE_THRESHOLD = 0.4  # Umbral de confianza

# --- Lista de Clases Maestra ---
# Lista definitiva y correcta para el modelo
classes = [
    "black_8",
    "blue_10",
    "blue_2",
    "dred_15",
    "dred_7",
    "green_14",
    "green_6",
    "orange_13",
    "orange_5",
    "purple_12",
    "purple_4",
    "red_11",
    "red_3",
    "white",
    "yellow_1",
    "yellow_9",
    "be_black_8",
    "be_blue_10",
    "be_blue_2",
    "be_dred_15",
    "be_dred_7",
    "be_green_14",
    "be_green_6",
    "be_purple_13",
    "be_purple_5",
    "be_pink_4",
    "be_pink_12",
    "be_red_11",
    "be_red_3",
    "be_white",
    "be_yellow_1",
    "be_yellow_9",
]

classes = [
    "black_8",  # solida negra 8, valida para 'black_8' y 'be_black_8'
    "blue_10",  # rayada azul y blanca 10, valida para 'blue_10'
    "blue_2",  # solida azul 2, valida para 'blue_2' y 'be_blue_2'
    "dred_15",  # rayada marron y blanca 15, valida para 'dred_15'
    "dred_7",  # solida marron 7, valida para 'dred_7' y 'be_dred_7'
    "green_14",  # rayada verde y blanca 14, valida para 'green_14'
    "green_6",  # solida verde 6, valida para 'green_6' y 'be_green_6'
    "orange_13",  # rayada naranja y blanca 13, valida para 'orange_13'
    "orange_5",  # solida naranja 5, valida para 'orange_5'
    "purple_12",  # rayada morada y blanca 12, valida para 'purple_12'
    "purple_4",  # solida morada 4, valida para 'purple_4'
    "red_11",  # rayada roja y blanca 11, valida para 'red_11'
    "red_3",  # solida roja 3, valida para 'red_3' y 'be_red_3'
    "white",  # solida blanca, valida para 'white' y 'be_white'
    "yellow_1",  # solida amarilla 1, valida para 'yellow_1' y 'be_yellow_1'
    "yellow_9",  # rayada amarilla y blanca 9, valida para 'yellow_9'
    "be_blue_10",  # rayada azul y negra 10, valida para 'be_blue_10'
    "be_dred_15",  # rayada morada y negra 15, valida para 'be_dred_15'
    "be_green_14",  # rayada verde y negra 14, valida para 'be_green_14'
    "be_purple_13",  # rayada morada y negra 13, valida para 'be_purple_13'
    "be_purple_5",  # solida morada 5, valida para 'be_purple_5'
    "be_pink_4",  # solida rosa 4, valida para 'be_pink_4'
    "be_pink_12",  # rayada rosa y negra 12, valida para 'be_pink_12'
    "be_red_11",  # rayada roja y negra 11, valida para 'be_red_11'
    "be_yellow_9",  # rayada amarilla y negra 9, valida para 'be_yellow_9'
]
# --- FIN DE LA CONFIGURACIÓN ---


def inferencia_en_lote():
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Cargar el modelo (una sola vez) ---
    try:
        model = YOLO(model_path)
        print(f"Modelo cargado desde: {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # --- 3. Bucle para procesar todas las imágenes ---
    image_files = [
        f
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No se encontraron imágenes en la carpeta: {TEST_IMAGES_DIR}")
        return

    print(f"\nIniciando inferencia en {len(image_files)} imágenes...")

    for image_filename in image_files:
        test_image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        print(f"\n--- Procesando: {image_filename} ---")

        # Realizar la inferencia en la imagen actual
        results = model.predict(
            source=test_image_path,
            save=False,
            imgsz=1024,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        # --- 4. Procesar y visualizar los resultados (para cada imagen) ---
        result = results[0]
        if not result.boxes:
            print("No se detectaron objetos en esta imagen.")
            continue

        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Detecciones en {image_filename}")

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = (
                classes[class_id]
                if class_id < len(classes)
                else f"ID_Desconocido_{class_id}"
            )

            # Dibujar caja
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Poner etiqueta
            text_y = y1 - 10 if y1 > 10 else y1 + 20
            ax.text(
                x1,
                text_y,
                f"{class_name} {confidence:.2f}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.7),
            )

            print(f"  Detectado: {class_name} (Confianza: {confidence:.2f})")

        plt.axis("off")

        # Guardar la imagen con un nombre único
        output_filename = f"resultado_{image_filename}"
        output_image_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
        print(f"  Imagen con detecciones guardada en: {output_image_path}")

        # Cerrar la figura para liberar memoria y evitar que se muestren todas al final
        plt.close(fig)

    print("\n\n--- Proceso de inferencia en bucle completado. ---")


if __name__ == "__main__":
    inferencia_en_lote()
