"""
Módulo de Detección Inicial de Bordes y Líneas (Fase 5, Paso 2).

Este script realiza las etapas de pre-procesamiento y feature extraction de Visión
por Computadora Clásica, que sirven como el "sensor" para el subsistema de Detección
de Mesa.

Propósito principal:
1.  Generar un mapa de bordes robusto (Canny).
2.  Extraer todas las líneas candidatas (Transformada de Hough) que serán
    filtradas posteriormente por el "cerebro" (filter_table_borders.py).
"""

import cv2
import numpy as np

# Carga la imagen
image_path = "detect_balls/tests/Black/test_pool_table_1.png"
# image_path = "detect_balls/tests/Black/test_pool_table_2.png"
# image_path = "detect_balls/tests/Black/test_pool_table_3.png"
# image_path = "detect_balls/tests/Classic/test_pool_table_4.jpg"
# image_path = "detect_balls/tests/Classic/test_pool_table_5.jpg"
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error al cargar la imagen: {image_path}")
    exit()

# Convertir a escala de grises (común para la detección de bordes)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Aplicar un desenfoque gaussiano para suavizar la imagen y reducir el ruido
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 1. Detección de Bordes con Canny
# Los umbrales (50 y 150) pueden necesitar ajuste
edges = cv2.Canny(blurred_image, 40, 125)
# edges = cv2.Canny(blurred_image, 50, 150)
cv2.imshow("Bordes Detectados (Canny)", edges)
cv2.waitKey(0)  # Presiona una tecla para continuar

# 2. Detección de Líneas con la Transformada de Hough
# cv2.HoughLinesP(imagen_bordes, rho, theta, umbral_interseccion, minLineLength, maxLineGap)
# rho: resolución de la distancia en píxeles
# theta: resolución del ángulo en radianes (np.pi/180 para 1 grado)
# umbral_interseccion: número mínimo de votos (intersecciones en el espacio de Hough)
# minLineLength: longitud mínima de una línea para ser considerada.
# maxLineGap: máxima brecha permitida entre puntos en la misma línea.
# Estos parámetros también necesitarán ajuste.
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=30, minLineLength=150, maxLineGap=20
)
# edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10

# Dibuja las líneas detectadas en la imagen original (o una copia)
image_with_lines = original_image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(
            image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2
        )  # Dibuja líneas rojas

cv2.imshow("Lineas Detectadas (Hough)", image_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
