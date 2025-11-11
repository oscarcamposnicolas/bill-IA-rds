# test_modelo_bucle.py
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN ---
# Mantén todas las variables que puedas querer cambiar aquí arriba

# --- Rutas del Proyecto ---
base_project_dir = os.path.join(".", "detect_pockets")
runs_dir = os.path.join(base_project_dir, "runs")
train_version = "detect_pockets_v12"

# Debes crear esta carpeta y colocar dentro todas las imágenes que quieras probar.
TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Classic")
# TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Black")

# Carpeta donde se guardarán los resultados de este script
output_dir = os.path.join(base_project_dir, "tests", "results_classic")
# output_dir = os.path.join(base_project_dir, "tests", "results_black")

# --- Ruta al Modelo ---
# Apunta al mejor modelo de tu entrenamiento final
model_path = os.path.join(
    runs_dir,
    train_version,
    "weights",
    "best.pt",
)

# --- Parámetros de Inferencia ---
CONFIDENCE_THRESHOLD = 0.4  # Umbral de confianza (ajústalo según necesites)

# --- Lista de Clases Maestra ---
# Asegúrate de que esta lista sea la definitiva y correcta para tu modelo
classes = ["pocket_corner", "pocket_side"]

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
