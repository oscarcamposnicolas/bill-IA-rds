# pre_etiquetado_definitivo.py
import os

from ultralytics import YOLO

# --- 1. CONFIGURACIÓN ---

# Ruta a tu mejor modelo entrenado hasta ahora
MODELO_PATH = "./detect_balls/runs/detect_balls_v12/weights/best.pt"

# Carpeta donde tienes TODAS tus nuevas imágenes "black edition" sin etiquetar
IMAGENES_NUEVAS_DIR = "./detect_balls/datasets/black_edition_raw/images/"

# Carpeta donde se guardarán las etiquetas generadas por la IA
ETIQUETAS_GENERADAS_DIR = "./detect_balls/datasets/black_edition_raw/labels/"

# Umbral de confianza
UMBRAL_CONFIANZA = 0.25

# ¡¡CRUCIAL!! Esta lista y su orden deben ser IDÉNTICOS a los del `data.yaml`
# con el que se entrenó el modelo que estás usando en MODELO_PATH.
CLASSES = [
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
]

CLASES_MAESTRA = [
    # --- Set Original (IDs 0-15) ---
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
    # --- Set Black Edition (IDs 16-31) ---
    "be_black_8",
    "be_blue_10",
    "be_blue_2",
    "be_dred_15",
    "be_dred_7",
    "be_green_14",
    "be_green_6",
    "be_purple_13",
    "be_purple_5",
    "be_pink_12",
    "be_pink_4",
    "be_red_11",
    "be_red_3",
    "be_white",
    "be_yellow_1",
    "be_yellow_9",
]

# --- FIN DE LA CONFIGURACIÓN ---

os.makedirs(ETIQUETAS_GENERADAS_DIR, exist_ok=True)

try:
    model = YOLO(MODELO_PATH)
    print(f"Modelo cargado exitosamente desde: {MODELO_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

try:
    lista_imagenes = [
        f
        for f in os.listdir(IMAGENES_NUEVAS_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not lista_imagenes:
        print(f"No se encontraron imágenes en: {IMAGENES_NUEVAS_DIR}")
        exit()
    print(f"Se encontraron {len(lista_imagenes)} imágenes para pre-etiquetar.")
except FileNotFoundError:
    print(f"Error: El directorio de imágenes no existe: {IMAGENES_NUEVAS_DIR}")
    exit()

# Procesar cada imagen
for nombre_imagen in lista_imagenes:
    ruta_imagen = os.path.join(IMAGENES_NUEVAS_DIR, nombre_imagen)
    print(f"\nProcesando imagen: {nombre_imagen}...")

    results = model.predict(ruta_imagen, conf=UMBRAL_CONFIANZA, verbose=False)
    result = results[0]

    nombre_base = os.path.splitext(nombre_imagen)[0]
    ruta_etiqueta = os.path.join(ETIQUETAS_GENERADAS_DIR, f"{nombre_base}.txt")

    with open(ruta_etiqueta, "w") as f_etiqueta:
        if result.boxes:
            print(f"  > Se detectaron {len(result.boxes)} objetos:")
            for box in result.boxes:
                coords_xywhn = box.xywhn[0]
                id_clase = int(box.cls[0])
                confianza = box.conf[0]

                nombre_clase = (
                    CLASSES[id_clase]
                    if id_clase < len(CLASSES)
                    else f"ID_DESCONOCIDO_{id_clase}"
                )
                print(
                    f"    - Propuesta: {nombre_clase} (ID: {id_clase}, Conf: {confianza:.2f})"
                )

                linea = f"{id_clase} {coords_xywhn[0]:.6f} {coords_xywhn[1]:.6f} {coords_xywhn[2]:.6f} {coords_xywhn[3]:.6f}\n"
                f_etiqueta.write(linea)
        else:
            print("  > No se detectó ningún objeto con la confianza suficiente.")

print("\n¡Proceso de pre-etiquetado completado!")
print(
    f"Las etiquetas generadas (con los IDs correctos) están en: {ETIQUETAS_GENERADAS_DIR}"
)
