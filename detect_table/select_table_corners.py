"""
Módulo de Selección Manual de Esquinas (Fase 5, Paso 1 - Herramienta de Validación).

Este script de utilidad permite la selección interactiva y manual de las cuatro
esquinas de la mesa de billar en una imagen.

Propósito principal:
1.  Validación Inicial: Probar el cálculo de la Matriz de Homografía con un
    input de coordenadas garantizadas, antes de integrar el detector automático
    (filter_table_borders.py).
2.  Generación de Datos: Obtener los puntos de origen (source points) para testear
    las funciones de Homografía de forma aislada.
"""

import cv2
import numpy as np

# Lista para almacenar los puntos de las esquinas seleccionadas
corner_points = []
current_image = None


def select_corners_mouse_callback(event, x, y, flags, param):
    global corner_points, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_points) < 4:
            # Dibuja un círculo en el punto seleccionado
            cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Selecciona Esquinas", current_image)

            corner_points.append((x, y))
            print(f"Punto {len(corner_points)} seleccionado: ({x}, {y})")
            if len(corner_points) == 4:
                print(
                    "¡4 esquinas seleccionadas! Presiona cualquier tecla para continuar."
                )
        else:
            print("Ya has seleccionado 4 esquinas. Presiona una tecla.")


def get_manual_corners(image_path):
    global corner_points, current_image
    corner_points = []  # Resetea los puntos para una nueva imagen

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return None

    current_image = (
        original_image.copy()
    )  # Trabajamos sobre una copia para no alterar la original

    cv2.namedWindow("Selecciona Esquinas")
    cv2.setMouseCallback("Selecciona Esquinas", select_corners_mouse_callback)

    print(
        "Por favor, haz clic en las 4 esquinas de la mesa en la imagen (ej: superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda)."
    )
    cv2.imshow("Selecciona Esquinas", current_image)

    # Espera hasta que se presione una tecla y se hayan seleccionado 4 puntos
    while not (len(corner_points) == 4 and cv2.waitKey(0) != -1):
        if cv2.waitKey(1) & 0xFF == 27:  # Si se presiona ESC
            print("Selección cancelada.")
            cv2.destroyAllWindows()
            return None
        if (
            cv2.getWindowProperty("Selecciona Esquinas", cv2.WND_PROP_VISIBLE) < 1
        ):  # Si se cierra la ventana
            print("Ventana cerrada. Selección cancelada.")
            return None

    cv2.destroyAllWindows()

    if len(corner_points) == 4:
        print("Esquinas seleccionadas:", corner_points)
        return np.array(corner_points, dtype="float32")
    else:
        print("No se seleccionaron 4 esquinas.")
        return None


# --- Uso ---
if __name__ == "__main__":
    image_file = "detect_table/tests/test_pool_table_1.png"
    selected_corners = get_manual_corners(image_file)

    if selected_corners is not None:
        # Aquí 'selected_corners' es un array de NumPy con los 4 puntos.
        # Estos son los "Puntos de Origen" para la homografía.
        print("\nCoordenadas de las esquinas en la imagen (Puntos de Origen):")
        print(selected_corners)

        # El siguiente paso sería definir los "Puntos de Destino" y calcular la homografía.
    else:
        print("No se pudo completar la selección de esquinas.")
