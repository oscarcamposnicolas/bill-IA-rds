# procesador_final.py
import json
import os
import random
from urllib.parse import unquote

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---

# ¡¡¡PASO MÁS IMPORTANTE!!!
# REEMPLAZA ESTA LISTA CON TU LISTA DE CLASES, EN EL ORDEN EXACTO EN QUE APARECE EN LABEL STUDIO.
CLASES_MAESTRA = [
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
    # Ejemplo de cómo podrían seguir las clases 'be_'
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

# Ruta al JSON exportado desde Label Studio
LABEL_STUDIO_JSON_PATH = "Proyecto_Bolas_LS/ls-export.json"

# Ruta a la carpeta que contiene las imágenes originales que usaste en Label Studio
IMAGENES_DIR = "Proyecto_Bolas_LS/imagenes/"

# Carpeta de salida para las etiquetas YOLO (.txt)
YOLO_LABELS_OUTPUT_DIR = "Proyecto_Bolas_LS/final_yolo_labels/"

# Carpeta de salida para las imágenes de verificación
VERIFICATION_DIR = "Proyecto_Bolas_LS/verification_results/"

# --- FIN DE LA CONFIGURACIÓN ---


def convertir_y_verificar():
    # --- FASE 1: CONVERSIÓN DE LS-JSON A YOLO-TXT ---
    print("--- INICIANDO FASE 1: Conversión de Label Studio a formato YOLO ---")

    os.makedirs(YOLO_LABELS_OUTPUT_DIR, exist_ok=True)
    class_to_id = {name: i for i, name in enumerate(CLASES_MAESTRA)}

    with open(LABEL_STUDIO_JSON_PATH, "r") as f:
        data = json.load(f)

    for task in data:
        image_path = task["image"]
        image_filename = os.path.basename(unquote(image_path))

        # Limpieza del hash de LS
        parts = image_filename.split("-", 1)
        if len(parts) > 1 and len(parts[0]) != 1:
            image_filename = parts[1]

        yolo_txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        output_path = os.path.join(YOLO_LABELS_OUTPUT_DIR, yolo_txt_filename)

        with open(output_path, "w") as f_yolo:
            if "label" in task and task["label"]:
                for annotation in task["label"]:
                    class_name = annotation["rectanglelabels"][0]
                    if class_name in class_to_id:
                        class_id = class_to_id[class_name]
                        x_ls, y_ls = annotation["x"], annotation["y"]
                        w_ls, h_ls = annotation["width"], annotation["height"]

                        x_norm, y_norm = x_ls / 100.0, y_ls / 100.0
                        w_norm, h_norm = w_ls / 100.0, h_ls / 100.0

                        x_center = x_norm + (w_norm / 2)
                        y_center = y_norm + (h_norm / 2)

                        f_yolo.write(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                        )

    print("--- FASE 1 COMPLETADA: Se han generado los archivos .txt ---")

    # --- FASE 2: VERIFICACIÓN VISUAL ---
    print("\n--- INICIANDO FASE 2: Verificación visual de las etiquetas generadas ---")
    os.makedirs(VERIFICATION_DIR, exist_ok=True)

    label_files = os.listdir(YOLO_LABELS_OUTPUT_DIR)
    archivos_a_verificar = random.sample(label_files, min(5, len(label_files)))

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

        label_path = os.path.join(YOLO_LABELS_OUTPUT_DIR, label_file)
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
    convertir_y_verificar()
