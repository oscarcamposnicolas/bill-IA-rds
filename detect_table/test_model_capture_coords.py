import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# PASO 0: CONFIGURACIÓN INICIAL
# -----------------------------------------------------------------------------

# Tu MATRIZ DE HOMOGRAFÍA (Calculada en el paso anterior)
# Asegúrate de que esta es la matriz correcta que calculaste para la imagen
# que vas a procesar o para una configuración de cámara similar.
H_matrix = np.array(
    [
        [9.01494445e-01, 1.69172272e-17, -8.56419723e01],
        [-8.24671937e-04, 9.16210522e-01, -8.23806031e01],
        [1.62899797e-06, 6.57494777e-06, 1.00000000e00],
    ]
)

# Ruta a tu modelo YOLO entrenado (el archivo .pt)
MODEL_PATH = "poolballs36.pt"  # Cambia esto
# Ejemplo: MODEL_PATH = 'runs/detect/train/weights/best.pt'

# Nombres de las clases: Deben estar en el MISMO ORDEN que usó tu modelo para entrenar.
# Generalmente este orden viene del archivo .yaml que usaste para el entrenamiento.
CLASS_NAMES = [
    "white",
    "blue_10",
    "dred_15",
    "black_8",
    "purple_12",
    "dred_7",
    "orange_13",
    "blue_2",
    "red_3",
    "green_6",
    "green_14",
    "red_11",
    "yellow_1",
    "orange_5",
    "yellow_9",
]  # ¡¡AJUSTA ESTA LISTA A TUS CLASES Y SU ORDEN CORRECTO!!

# Ruta a la imagen que quieres procesar
IMAGE_PATH = "tests/test_pool_table_1.png"  # Cambia esto
# Idealmente, usa la misma imagen para la cual seleccionaste las esquinas manualmente,
# o una tomada desde una perspectiva muy similar.

# -----------------------------------------------------------------------------
# PASO 1: CARGAR MODELO Y REALIZAR INFERENCIA
# -----------------------------------------------------------------------------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error al cargar el modelo YOLO: {e}")
    exit()

# Cargar la imagen original con OpenCV
img_original_cv = cv2.imread(IMAGE_PATH)
if img_original_cv is None:
    print(f"Error: No se pudo cargar la imagen {IMAGE_PATH}")
    exit()

# Realizar la inferencia con el modelo YOLO
print(f"Realizando inferencia en {IMAGE_PATH}...")
results = model(img_original_cv)  # La inferencia devuelve una lista de objetos Results

# Lista para guardar la información de las bolas con coordenadas transformadas
detected_balls_on_table_plane = []

# -----------------------------------------------------------------------------
# PASO 2: PROCESAR DETECCIONES Y TRANSFORMAR COORDENADAS
# -----------------------------------------------------------------------------
# 'results' es una lista, generalmente con un solo elemento si procesas una imagen.
if results and len(results) > 0:
    result = results[0]  # Tomamos el primer (y único) resultado
    boxes = (
        result.boxes
    )  # Accedemos al atributo 'boxes' que contiene la información de detección

    print(f"Se detectaron {len(boxes)} objetos.")

    for i in range(len(boxes)):
        # Obtener coordenadas del bounding box (formato xyxy)
        xyxy = boxes.xyxy[i].cpu().numpy()  # [xmin, ymin, xmax, ymax]

        # Obtener confianza y ID de clase
        confidence = boxes.conf[i].cpu().numpy()
        class_id = int(boxes.cls[i].cpu().numpy())

        # Obtener el nombre de la clase
        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"ClaseDesconocida_{class_id}"

        # A. Calcular el centro del bounding box en coordenadas de la IMAGEN
        xmin, ymin, xmax, ymax = xyxy
        center_x_image = (xmin + xmax) / 2
        center_y_image = (ymin + ymax) / 2

        # B. Preparar el punto para la transformación de perspectiva
        # El formato debe ser (1, 1, 2) para un solo punto, y de tipo float32
        point_in_image_to_transform = np.array(
            [[[center_x_image, center_y_image]]], dtype="float32"
        )

        # C. Aplicar la transformación de perspectiva usando la matriz H
        transformed_point_on_table = cv2.perspectiveTransform(
            point_in_image_to_transform, H_matrix
        )

        if transformed_point_on_table is not None:
            # Extraer las coordenadas transformadas (x_mesa, y_mesa)
            center_x_table = transformed_point_on_table[0][0][0]
            center_y_table = transformed_point_on_table[0][0][1]

            print(f"\n  Bola: {class_name} (Confianza: {confidence:.2f})")
            print(
                f"    Centro en Imagen (px): ({center_x_image:.1f}, {center_y_image:.1f})"
            )
            print(
                f"    Centro en Mesa (units): ({center_x_table:.1f}, {center_y_table:.1f})"
            )

            # Guardar la información
            detected_balls_on_table_plane.append(
                {
                    "clase": class_name,
                    "confianza": float(confidence),
                    "centro_imagen_px": (
                        round(center_x_image, 1),
                        round(center_y_image, 1),
                    ),
                    "centro_mesa_units": (
                        round(center_x_table, 1),
                        round(center_y_table, 1),
                    ),
                }
            )
        else:
            print(f"Error al transformar el punto para la detección de {class_name}.")
else:
    print("No se obtuvieron resultados de la inferencia.")

print("\n--- Proceso de Detección y Transformación Completado ---")
print(f"Bolas procesadas y transformadas: {len(detected_balls_on_table_plane)}")
for ball_data in detected_balls_on_table_plane:
    print(ball_data)

# -----------------------------------------------------------------------------
# OPCIONAL: MOSTRAR LA IMAGEN CON LAS DETECCIONES ORIGINALES (DE YOLO)
# -----------------------------------------------------------------------------
# Puedes descomentar estas líneas para ver los bounding boxes que YOLO dibuja
# en la imagen original. Esto es útil para verificar que YOLO está detectando
# las bolas correctamente antes de la transformación.

annotated_frame = results[0].plot()  # El método plot() dibuja las detecciones
cv2.imshow("Detecciones YOLO en Imagen Original", annotated_frame)
cv2.waitKey(0)  # Espera a que se presione una tecla
cv2.destroyAllWindows()
