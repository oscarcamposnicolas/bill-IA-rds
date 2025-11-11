import os
import shutil

import pandas as pd

# --- 1. Definir rutas y nombres de archivos ---
# Ruta base donde se encuentra la carpeta 'data'
base_project_dir = "."

# Directorios de origen de las imágenes y anotaciones CSV
data_base_dir = os.path.join(base_project_dir, "data", "BallsDataset")
train_src_dir = os.path.join(data_base_dir, "train")
test_src_dir = os.path.join(
    data_base_dir, "test"
)  # Usaremos test como 'val' si no hay una carpeta 'valid'
val_src_dir = os.path.join(
    data_base_dir, "valid"
)  # Si existe una carpeta 'valid' separada

# Directorios de destino para el formato YOLO
output_images_dir = os.path.join(base_project_dir, "data", "images")
output_labels_dir = os.path.join(base_project_dir, "data", "labels")

output_train_images_dir = os.path.join(output_images_dir, "train")
output_train_labels_dir = os.path.join(output_labels_dir, "train")
output_val_images_dir = os.path.join(output_images_dir, "val")
output_val_labels_dir = os.path.join(output_labels_dir, "val")
output_test_images_dir = os.path.join(
    output_images_dir, "test"
)  # Para el conjunto de test
output_test_labels_dir = os.path.join(
    output_labels_dir, "test"
)  # Para el conjunto de test

# Clases del dataset (ordenadas, esto define el class_id)
# Es CRÍTICO que el orden de estas clases sea CONSISTENTE y completo
classes = [
    "white",
    "yellow_1",
    "blue_2",
    "red_3",
    "purple_4",
    "orange_5",
    "green_6",
    "dred_7",
    "black_8",
    "yellow_9",
    "blue_10",
    "red_11",
    "purple_12",
    "orange_13",
    "green_14",
    "dred_15",
]

# Crear el mapeo de nombres de clase a IDs
class_to_id = {name: i for i, name in enumerate(classes)}


# --- 2. Función para procesar un conjunto de datos (train, val, test) ---
def process_dataset_split(src_dir, dest_images_dir, dest_labels_dir):
    """
    Procesa un directorio de origen de imágenes y CSV de anotaciones,
    copiando imágenes y generando archivos de anotación YOLO.
    """
    print(f"\nProcesando directorio: {src_dir}")
    # Crear directorios de destino si no existen
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)

    annotations_file_path = os.path.join(src_dir, "_annotations.csv")

    if not os.path.exists(annotations_file_path):
        print(
            f"Advertencia: No se encontró _annotations.csv en {src_dir}. Saltando este directorio."
        )
        return

    df_annotations = pd.read_csv(annotations_file_path)

    # Limpiar archivos .txt antiguos en el directorio de labels de destino
    for f in os.listdir(dest_labels_dir):
        if f.endswith(".txt"):
            os.remove(os.path.join(dest_labels_dir, f))

    for index, row in df_annotations.iterrows():
        img_filename = row["filename"]
        img_width = row["width"]
        img_height = row["height"]
        obj_class_name = row["class"]
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]

        # Verificar si la clase existe en nuestro mapeo
        if obj_class_name not in class_to_id:
            print(
                f"Advertencia: Clase '{obj_class_name}' no encontrada en la lista de clases definidas. Saltando anotación para {img_filename}."
            )
            continue

        # Copiar la imagen al directorio correspondiente (si no está ya allí)
        source_img_path = os.path.join(src_dir, img_filename)
        destination_img_path = os.path.join(dest_images_dir, img_filename)
        if not os.path.exists(destination_img_path):
            try:
                shutil.copy(source_img_path, destination_img_path)
            except FileNotFoundError:
                print(
                    f"Error: Imagen original no encontrada en {source_img_path}. Asegúrate de que las imágenes estén en el directorio '{src_dir}'."
                )
                continue  # Si la imagen no existe, no podemos procesar su anotación.

        # Convertir coordenadas a formato YOLO
        obj_class_id = class_to_id[obj_class_name]
        center_x = (xmin + xmax) / 2 / img_width
        center_y = (ymin + ymax) / 2 / img_height
        bbox_width = (xmax - xmin) / img_width
        bbox_height = (ymax - ymin) / img_height

        # Abrir el archivo .txt de la etiqueta en modo 'a' (append) para añadir todas las anotaciones de la imagen
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_filepath = os.path.join(dest_labels_dir, label_filename)

        with open(label_filepath, "a") as f:
            f.write(
                f"{obj_class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
            )

    print(f"Procesamiento de {src_dir} completado.")


# --- 3. Procesar cada conjunto de datos ---
process_dataset_split(train_src_dir, output_train_images_dir, output_train_labels_dir)
process_dataset_split(val_src_dir, output_val_images_dir, output_val_labels_dir)
process_dataset_split(test_src_dir, output_test_images_dir, output_test_labels_dir)

print("\nTodos los conjuntos de datos han sido procesados.")

# --- 4. Crear el archivo custom_data.yaml (importante para YOLO) ---
# Se asume que la estructura de salida es data/images/train, data/images/val, data/images/test
data_yaml_content = f"""
path: {os.path.abspath(os.path.join(base_project_dir, 'data'))} # Ruta absoluta a la carpeta 'data'
train: images/train # Ruta relativa a las imágenes de entrenamiento
val: images/val   # Ruta relativa a las imágenes de validación
test: images/test # Ruta relativa a las imágenes de prueba

# Número de clases
nc: {len(classes)}

# Nombres de las clases
names: {classes}
"""

with open(os.path.join(base_project_dir, "custom_data.yaml"), "w") as f:
    f.write(data_yaml_content)

print(
    f"Archivo custom_data.yaml generado en {os.path.join(base_project_dir, 'custom_data.yaml')}"
)

print("\n¡Preparación del dataset completada!")
print("Puedes continuar con el entrenamiento de tu modelo YOLO.")
