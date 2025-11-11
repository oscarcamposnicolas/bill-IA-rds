import cv2
import numpy as np

# Tu Matriz de Homografía calculada
H_matrix = np.array(
    [
        [9.01494445e-01, 1.69172272e-17, -8.56419723e01],
        [-8.24671937e-04, 9.16210522e-01, -8.23806031e01],
        [1.62899797e-06, 6.57494777e-06, 1.00000000e00],
    ]
)

# Coordenadas de ejemplo de los centros de las bolas en la IMAGEN ORIGINAL (en píxeles)
# (Más adelante, estos vendrán de la salida de tu modelo YOLO)
image_ball_centers = np.array(
    [
        [[300.0, 350.0]],  # Centro de la Bola Blanca (ejemplo)
        [[800.0, 400.0]],  # Centro de la Bola Roja (ejemplo)
    ],
    dtype="float32",
)

# Asegúrate de que el array de puntos tiene la forma correcta (N, 1, 2) o (1, N, 2)
# Si tienes una lista simple de tuplas [(x1,y1), (x2,y2)], puedes hacer:
# points_to_transform = np.array([[(x1,y1)], [(x2,y2)]], dtype="float32")

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
        # Estos son los puntos (x_mesa, y_mesa) que usarías para Pygame, por ejemplo.
        # Recuerda que estarán en el rango de 0 a TABLE_VIEW_WIDTH-1 y 0 a TABLE_VIEW_HEIGHT-1.
else:
    print("Error durante la transformación de perspectiva.")
