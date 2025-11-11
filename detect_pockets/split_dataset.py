"""
Módulo de División Final del Dataset (Fase 6, Paso 4).

Este script realiza la división rigurosa del conjunto de datos de troneras
aumentado en los conjuntos de Entrenamiento, Validación y Prueba (Train/Valid/Test)
siguiendo los ratios 80/10/10.

Propósito principal:
1.  Garantizar la reproducibilidad del entrenamiento mediante la aleatorización.
2.  Crear la estructura de directorios estándar esperada por el framework YOLO.
3.  Asegurar que el modelo sea evaluado sobre datos que nunca ha visto (Conjunto Test).
"""

import os
import random
import shutil

# --- 1. CONFIGURACIÓN DE RUTAS ---

# Directorio de origen (donde están las 780 imágenes/etiquetas)
SOURCE_IMAGES_DIR = "detect_pockets/dataset_aumentado/images/"
SOURCE_LABELS_DIR = "detect_pockets/dataset_aumentado/labels/"

# Directorio de destino (la estructura final que YOLO espera)
BASE_OUTPUT_DIR = "detect_pockets/dataset_final_pockets/"

# Definir los ratios de división
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# TEST_RATIO se infiere (0.1)

# --- 2. CREACIÓN DE DIRECTORIOS ---

# Crear la estructura de carpetas (train/valid/test)
train_img_dir = os.path.join(BASE_OUTPUT_DIR, "images/train")
train_lbl_dir = os.path.join(BASE_OUTPUT_DIR, "labels/train")
valid_img_dir = os.path.join(BASE_OUTPUT_DIR, "images/valid")
valid_lbl_dir = os.path.join(BASE_OUTPUT_DIR, "labels/valid")
test_img_dir = os.path.join(BASE_OUTPUT_DIR, "images/test")
test_lbl_dir = os.path.join(BASE_OUTPUT_DIR, "labels/test")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(valid_img_dir, exist_ok=True)
os.makedirs(valid_lbl_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_lbl_dir, exist_ok=True)

print("Directorios de destino creados.")

# --- 3. LECTURA Y BARAJA DE ARCHIVOS ---

# Obtener todos los nombres de archivo de imagen (ej. 'img_aug_1.jpg')
try:
    all_image_files = [
        f
        for f in os.listdir(SOURCE_IMAGES_DIR)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
except FileNotFoundError:
    print(
        f"Error: No se encontró el directorio de imágenes de origen: {SOURCE_IMAGES_DIR}"
    )
    exit()

if not all_image_files:
    print(f"Error: No se encontraron imágenes en {SOURCE_IMAGES_DIR}")
    exit()

# Barajar la lista aleatoriamente
random.shuffle(all_image_files)

total_files = len(all_image_files)
print(f"Total de {total_files} archivos de imagen/etiqueta encontrados.")

# --- 4. CÁLCULO DE ÍNDICES DE DIVISIÓN ---

train_split_index = int(total_files * TRAIN_RATIO)
valid_split_index = int(total_files * (TRAIN_RATIO + VALID_RATIO))

# Repartir los archivos en listas
train_files = all_image_files[:train_split_index]
valid_files = all_image_files[train_split_index:valid_split_index]
test_files = all_image_files[valid_split_index:]  # El resto

print(
    f"División calculada: {len(train_files)} (Train), {len(valid_files)} (Valid), {len(test_files)} (Test)"
)

# --- 5. FUNCIÓN DE COPIA/MOVIMIENTO ---


def move_files(file_list, dest_img_dir, dest_lbl_dir):
    """Mueve el par .jpg y .txt a sus nuevas carpetas."""
    moved_count = 0
    for file_name in file_list:
        base_name = os.path.splitext(file_name)[0]
        label_name = base_name + ".txt"

        src_img_path = os.path.join(SOURCE_IMAGES_DIR, file_name)
        src_lbl_path = os.path.join(SOURCE_LABELS_DIR, label_name)

        dest_img_path = os.path.join(dest_img_dir, file_name)
        dest_lbl_path = os.path.join(dest_lbl_dir, label_name)

        # Verificar que ambos archivos existen antes de mover
        if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
            try:
                # Usamos shutil.move para mover los archivos
                shutil.move(src_img_path, dest_img_path)
                shutil.move(src_lbl_path, dest_lbl_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moviendo {file_name}: {e}")
        else:
            print(f"Advertencia: Falta el par de archivos para {base_name}. Saltando.")

    return moved_count


# --- 6. EJECUCIÓN DEL MOVIMIENTO ---

print("\nMoviendo archivos de ENTRENAMIENTO...")
move_files(train_files, train_img_dir, train_lbl_dir)

print("\nMoviendo archivos de VALIDACIÓN...")
move_files(valid_files, valid_img_dir, valid_lbl_dir)

print("\nMoviendo archivos de PRUEBA...")
move_files(test_files, test_img_dir, test_lbl_dir)

print("\n--- ¡División completada! ---")
print(f"Dataset final listo en: {BASE_OUTPUT_DIR}")
