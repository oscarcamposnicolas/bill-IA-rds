import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- 1. Definir rutas y parámetros ---
base_project_dir = os.path.join(".", "detect_balls")

train_version = "Supermodelo_Fase2_Final"
# train_version = "detect_balls_v"
version_run = ""
nombre_epoch = "Supermodelo_Fase2_Final"
path_runs = "runs"
path_test_results = "test_results"

# Ruta al modelo entrenado
model_path = os.path.join(
    base_project_dir,
    path_runs,
    train_version + version_run,
    "weights",
    "best.pt",
)

print("modelo: ", model_path)

"""
model_path = os.path.join(
    base_project_dir,
    "models_custom",
    "supermodel.pt",
)
"""


# nombre_imagen_test_1 = "87_jpg.rf.d7116def1f52d243e1296a65c37b0c4c.jpg"
# nombre_imagen_test_1 = "412_jpg.rf.b505e28c8901b5bb4ad7c848f2a5d68b.jpg"
# nombre_imagen_test_1 = "833_jpg.rf.10091a238702573667ad5d0ce98610c8.jpg"
nombre_imagen_test_1 = "50_jpg.rf.8af48c542670796a51fe342700a543ab.jpg"
# nombre_imagen_test_1 = "148_jpg.rf.dace67876bd0a52dd413ea5225637e57.jpg"
# nombre_imagen_test_1 = "745_jpg.rf.6e4a996c29238c014b0bb9eb4b4e82b5.jpg"
# nombre_imagen_test_1 = "2025-06-15 20-03-49_aug_11.jpg"

"""
test_image_path_1 = os.path.join(
    base_project_dir,
    "data",
    "test",
    "images",
    nombre_imagen_test_1,
)
"""

test_image_path_1 = os.path.join(
    base_project_dir,
    "dataset_unificado_aumentado",
    "images",
    "valid",
    nombre_imagen_test_1,
)


nombre_imagen_test_2 = "test_pool_table_1.png"
nombre_imagen_test_2 = "test_pool_table_2.png"
nombre_imagen_test_2 = "test_pool_table_3.png"
nombre_imagen_test_2 = "test_pool_table_4.jpg"
# nombre_imagen_test_2 = "test_pool_table_5.jpg"

test_image_path_2 = os.path.join(
    base_project_dir,
    "tests",
    nombre_imagen_test_2,
)

nombre_imagen_test = nombre_imagen_test_1
test_image_path = test_image_path_1

# Clases de tu dataset (debe coincidir con la lista usada en custom_data.yaml y en prepare_yolo_dataset.py)
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

# --- 2. Cargar el modelo ---
try:
    model = YOLO(model_path)
    print(f"Modelo cargado desde: {model_path}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print(
        "Asegúrate de que la ruta al archivo 'last.pt' sea correcta y que el entrenamiento se haya guardado."
    )
    exit()

# --- 3. Realizar la inferencia ---
print(f"Realizando inferencia en: {test_image_path}")
# La función predict devuelve un objeto Results.
# verbose=False para no imprimir los detalles de inferencia en consola,
# conf=0.25 es el umbral de confianza para mostrar detecciones (ajústalo si detecta demasiado o muy poco)
results = model.predict(
    source=test_image_path, save=False, imgsz=1024, conf=0.4, verbose=True
)

# --- 4. Procesar y visualizar los resultados ---
if results:
    # Solo tomamos el primer resultado (ya que estamos procesando una sola imagen)
    result = results[0]

    # Cargar la imagen original para visualización
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # OpenCV carga BGR, matplotlib espera RGB

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Detecciones del Modelo YOLO")

    # Extraer bounding boxes y clases
    for box in result.boxes:
        # box.xyxy: coordenadas en formato [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # box.conf: confianza de la detección
        confidence = box.conf[0]

        # box.cls: ID de la clase detectada
        class_id = int(box.cls[0])
        class_name = (
            classes[class_id]
            if class_id < len(classes)
            else f"Unknown_Class_{class_id}"
        )

        # Dibujar el bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # Poner la etiqueta de la clase y la confianza
        # Usamos coordenadas relativas para el texto para que no se salga de la imagen
        text_x = x1
        text_y = (
            y1 - 10 if y1 - 10 > 0 else y1 + 20
        )  # Ajusta la posición del texto para que sea visible
        ax.text(
            text_x,
            text_y,
            f"{class_name} {confidence:.2f}",
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

        # Imprimir coordenadas y clase por consola
        print(
            f"Detectado: {class_name} (Confianza: {confidence:.2f}) - Coordenadas: xmin={x1}, ymin={y1}, xmax={x2}, ymax={y2}"
        )
        print(f"   Centro: ({int((x1+x2)/2)}, {int((y1+y2)/2)})")

    plt.axis("off")  # Ocultar ejes

    output_image_path = os.path.join(
        base_project_dir, path_test_results, nombre_epoch + nombre_imagen_test
    )  # O el nombre que prefieras

    plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
    print(f"\nImagen con detecciones guardada en: {output_image_path}")

    plt.show()

    # Si quisieras las coordenadas normalizadas (0 a 1) para un uso posterior:
    # Puedes usar result.boxes.xyxyn para obtenerlas directamente.
    # Por ejemplo:
    # normalized_boxes = result.boxes.xyxyn
    # for i, bbox_norm in enumerate(normalized_boxes):
    #     cls_id = int(result.boxes.cls[i])
    #     conf = result.boxes.conf[i]
    #     center_x_norm = (bbox_norm[0] + bbox_norm[2]) / 2
    #     center_y_norm = (bbox_norm[1] + bbox_norm[3]) / 2
    #     width_norm = bbox_norm[2] - bbox_norm[0]
    #     height_norm = bbox_norm[3] - bbox_norm[1]
    #     print(f"Normalizado: {classes[cls_id]} (Conf: {conf:.2f}) - Centro({center_x_norm:.4f}, {center_y_norm:.4f}), W({width_norm:.4f}), H({height_norm:.4f})")

else:
    print("No se detectaron objetos en la imagen de prueba.")
