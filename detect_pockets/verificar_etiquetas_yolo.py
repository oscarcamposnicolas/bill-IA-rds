"""
Módulo de Verificación Visual de Anotaciones de Troneras (Fase 6, Paso 3).

Este script de Control de Calidad (QC) visualiza las imágenes y sus correspondientes
bounding boxes (bboxes) a partir de los archivos de etiqueta YOLO (.txt).

Propósito principal:
1.  Validación de Integridad: Confirmar que las coordenadas normalizadas se traducen
    correctamente a bboxes en la imagen para el nuevo dataset de troneras.
2.  Sanity Check: Asegurar que el proceso de aumentación no ha introducido bboxes
    malformadas o que han quedado fuera de la imagen.
3.  Inspección Humana: Permitir la revisión manual de la calidad de las anotaciones
    antes de iniciar el entrenamiento del modelo especializado.
"""

import os
import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---

# Directorio que contiene las imágenes del dataset "black edition"
IMAGENES_DIR = "detect_pockets/dataset_aumentado/images/"
print("Images: ", IMAGENES_DIR)

# Directorio que contiene los archivos .txt finales en formato YOLO
YOLO_LABELS_DIR = "detect_pockets/dataset_aumentado/labels/"
print("Labels: ", YOLO_LABELS_DIR)

# Directorio donde se guardarán las imágenes con las cajas dibujadas
VERIFICATION_DIR = "detect_pockets/images_verification_results/"
print("Verification: ", VERIFICATION_DIR)

# Lista de clases maestra. ¡DEBE SER IDÉNTICA A LA DEL SCRIPT ANTERIOR!
CLASES_MAESTRA = ["pocket_corner", "pocket_side"]
print("Clases_maestra: ", CLASES_MAESTRA)

# Número de imágenes aleatorias a verificar
NUM_IMAGENES_A_VERIFICAR = 15

# --- FIN DE LA CONFIGURACIÓN ---


def verificar_etiquetas():
    print("\n--- INICIANDO FASE 2: Verificación visual de las etiquetas generadas ---")
    os.makedirs(VERIFICATION_DIR, exist_ok=True)

    label_files = os.listdir(YOLO_LABELS_DIR)
    archivos_a_verificar = random.sample(label_files, NUM_IMAGENES_A_VERIFICAR)

    for label_file in archivos_a_verificar:
        base_name = os.path.splitext(label_file)[0]

        image_path_to_verify = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential_path = os.path.join(IMAGENES_DIR, base_name + ext)
            if os.path.exists(potential_path):
                image_path_to_verify = potential_path
                break

        if not image_path_to_verify:
            continue

        image = cv2.imread(image_path_to_verify)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Verificación de: {os.path.basename(image_path_to_verify)}")

        label_path = os.path.join(YOLO_LABELS_DIR, label_file)
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id, x_center, y_center, width, height = map(float, parts)

                box_w = width * w
                box_h = height * h
                x1 = (x_center * w) - (box_w / 2)
                y1 = (y_center * h) - (box_h / 2)

                class_name = CLASES_MAESTRA[
                    int(class_id)
                ]  # Usamos la misma lista maestra

                rect = patches.Rectangle(
                    (x1, y1),
                    box_w,
                    box_h,
                    linewidth=2,
                    edgecolor="cyan",
                    facecolor="none",
                )
                ax.add_patch(rect)
                plt.text(
                    x1,
                    y1 - 10,
                    class_name,
                    color="cyan",
                    fontsize=12,
                    bbox=dict(facecolor="black", alpha=0.7),
                )

        plt.axis("off")
        output_save_path = os.path.join(
            VERIFICATION_DIR, os.path.basename(image_path_to_verify)
        )
        plt.savefig(output_save_path, bbox_inches="tight", pad_inches=0)
        plt.show()
        print(f"Imagen de verificación guardada en: {output_save_path}")

    print("--- FASE 2 COMPLETADA ---")


if __name__ == "__main__":
    verificar_etiquetas()
