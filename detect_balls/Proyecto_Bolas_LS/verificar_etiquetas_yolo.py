# verificar_etiquetas_yolo.py
import os
import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---

# Directorio que contiene las imágenes del dataset "black edition"
# IMAGENES_DIR = "Proyecto_Bolas_LS/dataset_be_aumentado/images/"
IMAGENES_DIR = "detect_balls/dataset_unificado_aumentado/images/train/"
IMAGENES_DIR = "detect_balls/dataset_hibrido/images/train/"
print("Images: ", IMAGENES_DIR)

# Directorio que contiene los archivos .txt finales en formato YOLO
# YOLO_LABELS_DIR = "Proyecto_Bolas_LS/dataset_be_aumentado/labels/"
YOLO_LABELS_DIR = "detect_balls/dataset_unificado_aumentado/labels/train/"
YOLO_LABELS_DIR = "detect_balls/dataset_hibrido/labels/train/"
print("Labels: ", YOLO_LABELS_DIR)

# Directorio donde se guardarán las imágenes con las cajas dibujadas
# VERIFICATION_DIR = "Proyecto_Bolas_LS/dataset_be_aumentado/verification_results/"
VERIFICATION_DIR = (
    "detect_balls/dataset_unificado_aumentado/images/verification_results/"
)
VERIFICATION_DIR = "detect_balls/dataset_hibrido/images/verification_results/"
print("Verification: ", VERIFICATION_DIR)

# YOLO_LABELS_OUTPUT_DIR = "Proyecto_Bolas_LS/final_yolo_labels/"

# Tu lista de clases maestra. ¡DEBE SER IDÉNTICA A LA DEL SCRIPT ANTERIOR!
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
CLASES_MAESTRA = [
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
print("Clases_maestra: ", CLASES_MAESTRA)

# Número de imágenes aleatorias que quieres verificar
NUM_IMAGENES_A_VERIFICAR = 5

# --- FIN DE LA CONFIGURACIÓN ---


def verificar_etiquetas():
    print("\n--- INICIANDO FASE 2: Verificación visual de las etiquetas generadas ---")
    os.makedirs(VERIFICATION_DIR, exist_ok=True)

    label_files = os.listdir(YOLO_LABELS_DIR)

    # print("Total files: ", len(label_files))

    # archivos_a_verificar = random.sample(label_files, min(5, len(label_files)))
    archivos_a_verificar = random.sample(label_files, NUM_IMAGENES_A_VERIFICAR)

    # print(archivos_a_verificar)

    for label_file in archivos_a_verificar:
        base_name = os.path.splitext(label_file)[0]

        # print(base_name)

        image_path_to_verify = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential_path = os.path.join(IMAGENES_DIR, base_name + ext)
            # print(potential_path)
            if os.path.exists(potential_path):
                image_path_to_verify = potential_path
                break

        # print(image_path_to_verify)

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
