import itertools
import math

import cv2
import numpy as np


def encontrar_punto_interseccion(line1_coords, line2_coords):
    """Calcula el punto de intersección de dos líneas."""
    x1, y1, x2, y2 = line1_coords
    x3, y3, x4, y4 = line2_coords

    A1, B1 = y2 - y1, x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2, B2 = y4 - y3, x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1

    if det == 0:
        return None

    px = (B2 * C1 - B1 * C2) / det
    py = (A1 * C2 - A2 * C1) / det

    return (int(px), int(py))


def ordenar_esquinas(esquinas):
    """Ordena las 4 esquinas en el orden canónico: TL, TR, BR, BL."""
    esquinas = np.array(esquinas)
    esquinas = esquinas[esquinas[:, 1].argsort()]

    top_esquinas = esquinas[:2]
    bottom_esquinas = esquinas[2:]

    # Ordenar por X: [TL, TR] y [BL, BR]
    top_esquinas = top_esquinas[top_esquinas[:, 0].argsort()]
    bottom_esquinas = bottom_esquinas[bottom_esquinas[:, 0].argsort()]

    tl, tr = top_esquinas[0], top_esquinas[1]
    bl, br = bottom_esquinas[0], bottom_esquinas[1]

    # Orden final para Homografía: TL, TR, BR, BL
    return [tl, tr, br, bl]


def score_quadrilateral(puntos_esquina, image_area):
    """Puntúa un cuadrilátero basado en su área y proporción (ideal 2:1)."""

    # ... (Métrica de Área - Implementación simplificada del área Shoelace) ...
    points_array = np.array(puntos_esquina)
    area = 0.5 * np.abs(
        np.dot(points_array[:, 0], np.roll(points_array[:, 1], 1))
        - np.dot(points_array[:, 1], np.roll(points_array[:, 0], 1))
    )

    if area < (image_area * 0.05) or area > (image_area * 0.95):
        return 0.0  # Descartar áreas muy pequeñas o muy grandes

    # Métrica de Proporción (Ancho promedio / Alto promedio)
    width_top = np.linalg.norm(points_array[1] - points_array[0])
    width_bottom = np.linalg.norm(points_array[2] - points_array[3])
    height_left = np.linalg.norm(points_array[3] - points_array[0])
    height_right = np.linalg.norm(points_array[2] - points_array[1])

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2

    if avg_height == 0:
        return 0.0

    aspect_ratio = avg_width / avg_height
    IDEAL_RATIO = 2.0

    # Penalización por error de proporción (ej. un 25% de error en la proporción)
    ratio_error_factor = abs(aspect_ratio - IDEAL_RATIO) / IDEAL_RATIO
    ratio_score = max(0, 1.0 - ratio_error_factor)

    # Puntuación final: Priorizamos el área y la buena proporción.
    return area * ratio_score


def find_best_quadrilateral(horizontal_lines, vertical_lines, image_shape):
    """
    Busca la mejor combinación de 2 líneas H y 2 líneas V que forma un rectángulo.
    Las líneas vienen en formato (coords, length).
    """
    best_score = 0.0
    best_lines = None
    image_area = image_shape[0] * image_shape[1]

    # Iterar todas las combinaciones de 2 líneas horizontales y 2 verticales
    for combo_h in itertools.combinations(horizontal_lines, 2):
        for combo_v in itertools.combinations(vertical_lines, 2):

            # Extraer solo las coordenadas (los coords son el elemento [0] de la tupla)
            h_coords = [c[0] for c in combo_h]
            v_coords = [c[0] for c in combo_v]

            # Calcular las 4 intersecciones
            raw_corners = [
                encontrar_punto_interseccion(h_coords[0], v_coords[0]),
                encontrar_punto_interseccion(h_coords[0], v_coords[1]),
                encontrar_punto_interseccion(h_coords[1], v_coords[0]),
                encontrar_punto_interseccion(h_coords[1], v_coords[1]),
            ]

            valid_corners = [p for p in raw_corners if p is not None]

            if len(valid_corners) == 4:
                # Ordenar las esquinas para el cálculo de la puntuación
                ordered_corners = ordenar_esquinas(valid_corners)

                # Puntuar el cuadrilátero
                score = score_quadrilateral(ordered_corners, image_area)

                if score > best_score:
                    best_score = score
                    # Guardamos las 4 líneas originales que generaron la mejor puntuación
                    best_lines = h_coords + v_coords

    # Retornar las 4 líneas seleccionadas (que luego se usarán para la Homografía)
    return best_lines, best_score


# Carga la imagen
image_path = "tests/test_pool_table_1.png"  # La misma que antes
image_path = "detect_balls/tests/Black/test_pool_table_1.png"

original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error al cargar la imagen: {image_path}")
    exit()

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred_image, 40, 125)  # Ajusta estos umbrales si es necesario

# Detección de Líneas con la Transformada de Hough (Probabilística)
# Ajusta threshold, minLineLength, maxLineGap según tu experimentación anterior
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=30, minLineLength=150, maxLineGap=20
)

image_with_filtered_lines = original_image.copy()
horizontal_lines = []
vertical_lines = []

if lines is not None:
    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]

        # Calcular longitud de la línea
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calcular ángulo de la línea
        if (x2 - x1) == 0:  # Línea vertical
            angle = 90.0
        else:
            angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))

        # --- Criterios de Filtrado (¡NECESITARÁS AJUSTAR ESTO!) ---
        # 1. Filtrar por longitud mínima
        MIN_LINE_LENGTH = 100  # Ejemplo: solo líneas de más de 100 píxeles
        if length < MIN_LINE_LENGTH:
            continue

        # 2. Clasificar en horizontales y verticales (aproximadas)
        # Ángulos cercanos a 0° (horizontales), ángulos cercanos a +/-90° (verticales)
        ANGLE_THRESHOLD = (
            30  # Ejemplo: +/- 30 grados de la horizontal/vertical perfecta
        )

        if abs(angle) < ANGLE_THRESHOLD:  # Considerada "horizontal-ish"
            horizontal_lines.append(((x1, y1, x2, y2), length))
            # cv2.line(image_with_filtered_lines, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde para horizontales
        elif abs(abs(angle) - 90) < ANGLE_THRESHOLD:  # Considerada "vertical-ish"
            vertical_lines.append(((x1, y1, x2, y2), length))
            # cv2.line(image_with_filtered_lines, (x1, y1), (x2, y2), (255, 0, 0), 2) # Azul para verticales
        # else: # Líneas diagonales que no cumplen los umbrales (opcional)
        # cv2.line(image_with_filtered_lines, (x1, y1), (x2, y2), (0,0,0), 1) # Negro para otras

    # Opcional: Quedarse con las N líneas más largas de cada categoría
    # Por ejemplo, las 2 más largas horizontales y las 2 más largas verticales
    horizontal_lines.sort(
        key=lambda item: item[1], reverse=True
    )  # Ordenar por longitud
    vertical_lines.sort(key=lambda item: item[1], reverse=True)

    # Dibujar, por ejemplo, las 4 líneas horizontales y 4 verticales más largas (o menos si no hay tantas)
    # Esto es una simplificación; idealmente buscarías las líneas que mejor forman el rectángulo de la mesa

    # Dibujamos las líneas seleccionadas
    selected_lines_for_corners = []

    print(
        f"Líneas horizontales detectadas (filtradas por longitud > {MIN_LINE_LENGTH}): {len(horizontal_lines)}"
    )
    for i, (line_coords, length) in enumerate(
        horizontal_lines[:4]
    ):  # Tomar hasta las 4 más largas
        x1, y1, x2, y2 = line_coords
        cv2.line(
            image_with_filtered_lines, (x1, y1), (x2, y2), (0, 255, 0), 3
        )  # Verde más grueso
        selected_lines_for_corners.append(line_coords)
        print(f"  H{i+1}: ({x1},{y1})-({x2},{y2}), Longitud: {length:.0f}")

    print(
        f"Líneas verticales detectadas (filtradas por longitud > {MIN_LINE_LENGTH}): {len(vertical_lines)}"
    )
    for i, (line_coords, length) in enumerate(
        vertical_lines[:4]
    ):  # Tomar hasta las 4 más largas
        x1, y1, x2, y2 = line_coords
        cv2.line(
            image_with_filtered_lines, (x1, y1), (x2, y2), (255, 0, 0), 3
        )  # Azul más grueso
        selected_lines_for_corners.append(line_coords)
        print(f"  V{i+1}: ({x1},{y1})-({x2},{y2}), Longitud: {length:.0f}")

else:
    print("No se detectaron líneas con los parámetros actuales.")

cv2.imshow("Imagen Original", original_image)
cv2.imshow("Bordes (Canny)", edges)
cv2.imshow("Lineas Filtradas", image_with_filtered_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

# El siguiente paso sería tomar 'selected_lines_for_corners' y encontrar sus intersecciones.
