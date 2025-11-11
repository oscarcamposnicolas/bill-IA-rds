# generar_meta_dataset.py
import os

import pandas as pd
from tqdm import tqdm  # Para una bonita barra de progreso (pip install tqdm)
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN ---

# Rutas a los datasets
DATASET_HIBRIDO_DIR = "detect_balls/dataset_hibrido/"
DATASET_ORIGINAL_UNIFICADO_DIR = (
    "detect_balls/dataset_hibrido/"  # Necesario para el ground truth
)

# Ruta a tu mejor modelo híbrido
MODEL_PATH = "detect_balls/runs/Modelo_Hibrido_v1/weights/best.pt"

# Nombre del archivo de salida
OUTPUT_CSV_PATH = "RandomForestClassifier/meta_dataset.csv"

# Listas de clases que ya tenemos definidas
CLASES_HIBRIDAS = [
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
    "be_blue_10",
    "be_dred_15",
    "be_green_14",
    "be_purple_13",
    "be_purple_5",
    "be_pink_4",
    "be_pink_12",
    "be_red_11",
    "be_yellow_9",
]
DELATORES_BE = {
    "be_blue_10",
    "be_dred_15",
    "be_green_14",
    # "be_purple_13",
    # "be_purple_5",
    "be_pink_4",
    "be_pink_12",
    "be_red_11",
    "be_yellow_9",
}
DELATORES_CLASSIC = {
    "blue_10",
    "dred_15",
    "green_14",
    "orange_13",
    "orange_5",
    # "purple_12",
    # "purple_4",
    "red_11",
    "yellow_9",
}

# Lista original de 32 clases para determinar el target
CLASES_MAESTRA_ORIGINAL = [
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

# --- FIN DE LA CONFIGURACIÓN ---


def get_ground_truth_context(label_path_original):
    """
    Lee el archivo de etiqueta del dataset original (32 clases) para determinar el contexto real.
    Devuelve 1 si es 'black_edition', 0 si es 'classic'.
    """
    try:
        with open(label_path_original, "r") as f:
            for line in f:
                id_original = int(float(line.strip().split()[0]))
                nombre_clase_original = CLASES_MAESTRA_ORIGINAL[id_original]
                if nombre_clase_original.startswith("be_"):
                    return 1  # Es Black Edition
    except (IOError, IndexError, ValueError):
        return 0  # Si hay error o no se encuentra, asumimos classic
    return 0  # Si termina el bucle sin encontrar delatores, es Classic


def generar_dataset():
    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    all_features = []

    # Iterar sobre train, valid y test
    for split in ["train", "valid", "test"]:
        images_dir = os.path.join(DATASET_HIBRIDO_DIR, "images", split)
        labels_original_dir = os.path.join(
            DATASET_ORIGINAL_UNIFICADO_DIR, "labels", split
        )

        if not os.path.isdir(images_dir):
            continue

        print(f"\nProcesando el conjunto de datos: {split}")
        for image_filename in tqdm(os.listdir(images_dir)):
            if not image_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(images_dir, image_filename)
            base_name = os.path.splitext(image_filename)[0]

            # 1. Realizar predicción con el modelo híbrido
            results = model.predict(image_path, verbose=False, conf=0.25)
            result = results[0]

            # 2. Inicializar y extraer características
            puntuacion_be = 0.0
            puntuacion_classic = 0.0
            confianza_total = 0.0
            counts = {f"count_{cls}": 0 for cls in CLASES_HIBRIDAS}

            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    clase = CLASES_HIBRIDAS[class_id]
                    conf = float(box.conf[0])

                    counts[f"count_{clase}"] += 1
                    confianza_total += conf

                    if clase in DELATORES_BE:
                        puntuacion_be += conf
                    elif clase in DELATORES_CLASSIC:
                        puntuacion_classic += conf

            total_detecciones = len(result.boxes)
            confianza_media = (
                confianza_total / total_detecciones if total_detecciones > 0 else 0
            )

            # 3. Determinar el Ground Truth (el contexto real)
            label_original_path = os.path.join(labels_original_dir, base_name + ".txt")
            target = get_ground_truth_context(label_original_path)

            # 4. Guardar la fila de datos
            row_data = {
                "image_path": image_path,
                "puntuacion_be": puntuacion_be,
                "puntuacion_classic": puntuacion_classic,
                "total_detecciones": total_detecciones,
                "confianza_media": confianza_media,
                "target": target,  # 0 para classic, 1 para black_edition
            }
            row_data.update(counts)
            all_features.append(row_data)

    # 5. Crear y guardar el DataFrame
    print("\nCreando el archivo CSV final...")
    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"¡Éxito! Meta-dataset guardado en '{OUTPUT_CSV_PATH}' con {len(df)} filas.")
    print("\nResumen del dataset:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    # Necesitarás instalar pandas y tqdm si no los tienes
    # pip install pandas tqdm
    generar_dataset()
