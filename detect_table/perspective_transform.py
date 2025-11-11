import cv2
import numpy as np

# Puntos de origen (los que seleccionaste manualmente)
source_points = np.array(
    [
        [95.0, 90.0],  # Superior-Izquierda
        [1206.0, 91.0],  # Superior-Derecha
        [1210.0, 639.0],  # Inferior-Derecha
        [95.0, 637.0],  # Inferior-Izquierda
    ],
    dtype="float32",
)

# Puntos de destino (en la mesa ideal)
# Definamos un ancho y alto para nuestra mesa rectificada
TABLE_VIEW_WIDTH = 1000
TABLE_VIEW_HEIGHT = 500  # Manteniendo una proporción aproximada de 2:1

destination_points = np.array(
    [
        [0, 0],  # Superior-Izquierda
        [TABLE_VIEW_WIDTH - 1, 0],  # Superior-Derecha
        [TABLE_VIEW_WIDTH - 1, TABLE_VIEW_HEIGHT - 1],  # Inferior-Derecha
        [0, TABLE_VIEW_HEIGHT - 1],  # Inferior-Izquierda
    ],
    dtype="float32",
)

# Calcular la matriz de homografía
H_matrix, status = cv2.findHomography(source_points, destination_points)
# Alternativamente, y a menudo preferido para solo 4 puntos:
# H_matrix = cv2.getPerspectiveTransform(source_points, destination_points)


if H_matrix is not None:
    print("Matriz de Homografía calculada (H):")
    print(H_matrix)

    # ¿Qué podemos hacer ahora con esta matriz?
    # 1. Aplicar la transformación a toda la imagen para obtener una vista cenital (opcional, bueno para visualización)
    # 2. Transformar las coordenadas específicas de las bolas detectadas por YOLO (nuestro objetivo principal)

    # Ejemplo de cómo obtener la vista cenital de la mesa (Paso Opcional de Visualización)
    # Carga la imagen original de nuevo
    # image_path = 'ruta/a/tu/imagen_de_billar.jpg' # La misma que usaste antes
    # original_image = cv2.imread(image_path)
    # if original_image is not None:
    #     warped_table_view = cv2.warpPerspective(original_image, H_matrix, (TABLE_VIEW_WIDTH, TABLE_VIEW_HEIGHT))
    #     cv2.imshow("Imagen Original", original_image)
    #     cv2.imshow("Vista Cenital de la Mesa (Rectificada)", warped_table_view)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print(f"Error: No se pudo cargar la imagen original {image_path} para la rectificación.")

else:
    print("No se pudo calcular la matriz de homografía.")
