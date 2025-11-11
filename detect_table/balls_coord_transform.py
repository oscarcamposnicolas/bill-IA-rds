"""
Módulo de Conversión de Coordenadas (Fase 5, Paso 5).

Este script ejecuta la etapa final de la Visión por Computadora: la transformación
de coordenadas. Su propósito es mapear los centroides de las bolas detectadas
por YOLO (coordenadas de píxeles distorsionadas por la perspectiva) al sistema
de coordenadas plano y cenital de la mesa ideal (0 a 1000 en X, 0 a 500 en Y).

Este proceso es CRUCIAL, ya que convierte datos visuales en datos geométricos
analizables para la IA de juego.
"""

import cv2
import numpy as np

# Matriz de Homografía calculada
H_matrix = np.array(
    [
        [9.01494445e-01, 1.69172272e-17, -8.56419723e01],
        [-8.24671937e-04, 9.16210522e-01, -8.23806031e01],
        [1.62899797e-06, 6.57494777e-06, 1.00000000e00],
    ]
)

# Coordenadas de ejemplo de los centros de las bolas en la IMAGEN ORIGINAL (en píxeles)
# (Más adelante, estos vendrán de la salida del modelo YOLO)
image_ball_centers = np.array(
    [
        [[300.0, 350.0]],  # Centro de la Bola Blanca (ejemplo)
        [[800.0, 400.0]],  # Centro de la Bola Roja (ejemplo)
    ],
    dtype="float32",
)

if image_ball_centers.ndim == 2:  # Si es (N,2)
    # Necesita ser (N,1,2) para perspectiveTransform
    image_ball_centers = image_ball_centers.reshape(-1, 1, 2)

# Aplicar la transformación de perspectiva
table_ball_centers = cv2.perspectiveTransform(image_ball_centers, H_matrix)

if table_ball_centers is not None:
    print("Coordenadas de las bolas en la imagen original:")
    for i, point in enumerate(image_ball_centers):
        print(f"  Bola {i+1}: ({point[0][0]:.1f}, {point[0][1]:.1f})")

    print("\nCoordenadas de las bolas transformadas (en el plano de la mesa ideal):")
    for i, point in enumerate(table_ball_centers):
        print(f"  Bola {i+1}: ({point[0][0]:.1f}, {point[0][1]:.1f})")
        # Estos son los puntos (x_mesa, y_mesa) que usaríamos para Pygame, por ejemplo.
        # Recordar que estarán en el rango de 0 a TABLE_VIEW_WIDTH-1 y 0 a TABLE_VIEW_HEIGHT-1.
else:
    print("Error durante la transformación de perspectiva.")
