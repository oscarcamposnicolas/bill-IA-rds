import itertools
import math

import cv2
import numpy as np

# =========================================================================
# I. FUNCIONES MATEMÁTICAS AUXILIARES (EL NÚCLEO DE ROBUSTEZ)
# =========================================================================

'''
def encontrar_punto_interseccion(line1_coords, line2_coords):
    """Calcula el punto de intersección de dos líneas."""
    x1, y1, x2, y2 = line1_coords
    x3, y3, x4, y4 = line2_coords

    A1, B1 = y2 - y1, x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2, B2 = y4 - y3, x3 - x4
    C2 = A2 * x3 + B2 * y3
'''


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
        return None  # Paralelas o colineales

    px = (B2 * C1 - B1 * C2) / det
    py = (A1 * C2 - A2 * C1) / det

    # Devolvemos el punto redondeado
    return (int(px), int(py))


def ordenar_esquinas(esquinas):
    """Ordena las 4 esquinas en el orden canónico: TL, TR, BR, BL."""
    esquinas = np.array(esquinas)

    # 1. Ordenar por Y para separar superior e inferior
    esquinas = esquinas[esquinas[:, 1].argsort()]

    top_esquinas = esquinas[:2]
    bottom_esquinas = esquinas[2:]

    # 2. Ordenar Top y Bottom por X
    top_esquinas = top_esquinas[top_esquinas[:, 0].argsort()]
    bottom_esquinas = bottom_esquinas[bottom_esquinas[:, 0].argsort()]

    tl, tr = top_esquinas[0], top_esquinas[1]
    bl, br = bottom_esquinas[0], bottom_esquinas[1]

    # Orden final para Homografía: TL, TR, BR, BL
    return [tl, tr, br, bl]


def score_quadrilateral(puntos_esquina, image_area):
    """Puntúa un cuadrilátero basado en su área y proporción, con tolerancia."""

    points_array = np.array(puntos_esquina)

    # 1. Métrica de Área (usamos el área, pero eliminamos el filtro inicial estricto)
    area = 0.5 * np.abs(
        np.dot(points_array[:, 0], np.roll(points_array[:, 1], 1))
        - np.dot(points_array[:, 1], np.roll(points_array[:, 0], 1))
    )

    # Descartar cuadriláteros minúsculos (e.g., menos del 1% del área) o gigantes.
    if area < (image_area * 0.01) or area > (image_area * 1.5):
        return 0.0

    # 2. Métrica de Proporción (Ancho promedio / Alto promedio)
    # Ya que la mesa está en perspectiva, no penalizaremos el ratio 2:1 tan fuerte.
    width_top = np.linalg.norm(points_array[1] - points_array[0])
    # ... (cálculo de width_bottom, height_left, height_right, avg_width, avg_height) ...
    width_bottom = np.linalg.norm(points_array[2] - points_array[3])
    height_left = np.linalg.norm(points_array[3] - points_array[0])
    height_right = np.linalg.norm(points_array[2] - points_array[1])

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2

    if avg_height == 0:
        return 0.0
    aspect_ratio = avg_width / avg_height

    # ¡CLAVE!: Penalizar las proporciones absurdas.
    # Un cuadrilátero de billar visto en perspectiva aún tendrá una proporción
    # entre 0.8:1 y 4:1. Si es más extremo, es ruido.
    if aspect_ratio < 0.8 or aspect_ratio > 4.0:
        return 0.0

    # Suavizamos la penalización: Solo premiamos la pertenencia al rango.
    # Usamos una función simple de regresión para mantener el área como factor principal.

    # Final Score: Área * Puntuación de forma.
    # Multiplicamos el área por un factor que premia las esquinas cercanas a 90 grados.
    # (Para simplificar, usaremos el área, ya que el filtro de rango de proporción es suficiente para eliminar el ruido).

    # NUEVA Puntuación Simplificada (Más robusta contra la perspectiva):
    # La puntuación más alta es el área más grande dentro de un ratio razonable.
    return area


# Versión CORREGIDA de find_best_quadrilateral (Filtro por Ratio de Área)
def find_best_quadrilateral(horizontal_lines, vertical_lines, image_shape):
    """
    Busca la mejor combinación de 2 líneas H y 2 líneas V.
    Implementa la Heurística del "Más Interior" (propuesta por el alumno):
    1. Filtra todos los candidatos por un ÁREA MÍNIMA (para eliminar ruido).
    2. De los restantes, selecciona el que tenga el ÁREA MÁS PEQUEÑA (el más interior).
    """
    all_candidates = []
    image_area = image_shape[0] * image_shape[1]
    img_height, img_width, _ = image_shape

    # Iterar todas las combinaciones posibles
    for combo_h in itertools.combinations(horizontal_lines, 2):
        for combo_v in itertools.combinations(vertical_lines, 2):

            h_coords = [c[0] for c in combo_h]
            v_coords = [c[0] for c in combo_v]

            raw_corners = [
                encontrar_punto_interseccion(h_coords[0], v_coords[0]),
                encontrar_punto_interseccion(h_coords[0], v_coords[1]),
                encontrar_punto_interseccion(h_coords[1], v_coords[0]),
                encontrar_punto_interseccion(h_coords[1], v_coords[1]),
            ]

            valid_corners = [p for p in raw_corners if p is not None]

            if len(valid_corners) == 4:
                # Validación de Rango (para evitar fallos por coordenadas extremas)
                is_valid_range = True
                for x, y in valid_corners:
                    if (
                        x < -img_width
                        or x > 2 * img_width
                        or y < -img_height
                        or y > 2 * img_height
                    ):
                        is_valid_range = False
                        break

                if not is_valid_range:
                    continue

                ordered_corners = ordenar_esquinas(valid_corners)
                # La puntuación (score) es principalmente el área
                score = score_quadrilateral(ordered_corners, image_area)

                if score > 0.0:
                    all_candidates.append(
                        {
                            "score": score,  # 'score' es el área ponderada por la proporción
                            "lines": h_coords + v_coords,
                            "corners": ordered_corners,
                        }
                    )

    # --------------------------------------------------------
    # NUEVA LÓGICA FINAL: SELECCIÓN POR ÁREA MÍNIMA (EL MÁS INTERIOR)
    # --------------------------------------------------------
    if not all_candidates:
        return None, 0.0, None

    # 1. Filtro de Área Mínima (Tu restricción)
    # Descartamos ruido que sea menor al 10% del área de la imagen
    MIN_AREA_THRESHOLD = image_area * 0.10

    valid_candidates = [c for c in all_candidates if c["score"] > MIN_AREA_THRESHOLD]

    if not valid_candidates:
        print(
            "ADVERTENCIA: No se encontraron candidatos por encima del umbral de área mínima (10%)."
        )
        # Fallback: Usar el más grande de todos los candidatos (el mejor score)
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        best_fallback = all_candidates[0]
        return best_fallback["lines"], best_fallback["score"], best_fallback["corners"]

    # 2. Búsqueda del "Más Interior" (El de menor área)
    # Ordenamos los candidatos válidos por puntuación (área) ASCENDENTE
    valid_candidates.sort(key=lambda x: x["score"], reverse=False)

    # El primer elemento (índice 0) es ahora el más pequeño VÁLIDO
    best_interior_candidate = valid_candidates[0]

    print(
        f"DEBUG: ¡Selección Final por Heurística de Área Mínima (Más Interior)! Score: {best_interior_candidate['score']:.0f}"
    )
    return (
        best_interior_candidate["lines"],
        best_interior_candidate["score"],
        best_interior_candidate["corners"],
    )


def filter_by_color(original_image, lines_list, color_tolerance=35):
    """Filtra las líneas que pasan sobre la mesa, buscando que no sean marrones/oscuras."""
    filtered_lines = []

    # Definimos una zona de color oscuro/madera a evitar (B, G, R)
    # Por ejemplo, valores bajos en general, característicos de la madera oscura.
    MIN_COLOR_VALUE = 40

    for line_data in lines_list:
        line_coords, length = line_data
        x1, y1, x2, y2 = line_coords

        # Muestreamos puntos a lo largo de la línea (e.g., 5 puntos)
        num_samples = 5
        x = np.linspace(x1, x2, num_samples, dtype=int)
        y = np.linspace(y1, y2, num_samples, dtype=int)

        is_dark_or_wood = False

        for px, py in zip(x, y):
            # Asegurarse de que el punto está dentro de los límites de la imagen
            if 0 <= py < original_image.shape[0] and 0 <= px < original_image.shape[1]:

                # BGR del punto
                B, G, R = original_image[py, px]

                # Criterio: Si todos los componentes de color son bajos, es probablemente una línea de sombra/madera oscura.
                if B < MIN_COLOR_VALUE and G < MIN_COLOR_VALUE and R < MIN_COLOR_VALUE:
                    is_dark_or_wood = True
                    break

        # Solo conservamos las líneas que no cruzan zonas muy oscuras o de madera
        if not is_dark_or_wood:
            filtered_lines.append(line_data)

    return filtered_lines


# =========================================================================
# II. LÓGICA DE DIBUJO Y EJECUCIÓN PRINCIPAL
# =========================================================================

# Carga la imagen (MODIFICAR SEGÚN TU RUTA)
image_path = "detect_balls/tests/Black/test_pool_table_1.png"
# image_path = "detect_balls/tests/Black/test_pool_table_2.png"
# image_path = "detect_balls/tests/Black/test_pool_table_3.png"
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error al cargar la imagen: {image_path}")
    exit()

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Parámetros validados para máxima sensibilidad
edges = cv2.Canny(blurred_image, 40, 125)

# Detección de Líneas con los parámetros validados (threshold bajo para capturar todo)
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=30, minLineLength=150, maxLineGap=20
)

# Inicialización de listas
horizontal_lines = []
vertical_lines = []
image_with_filtered_lines = original_image.copy()

if lines is not None:
    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]

        # Calcular longitud
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calcular ángulo
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)

        MIN_LINE_LENGTH = 50  # 100
        ANGLE_THRESHOLD = 45  # +/- 45 grados

        if length < MIN_LINE_LENGTH:
            continue

        # Clasificación H/V
        if abs(angle_deg) < ANGLE_THRESHOLD or abs(angle_deg) > (180 - ANGLE_THRESHOLD):
            # Línea horizontal
            horizontal_lines.append(((x1, y1, x2, y2), length))
        elif abs(abs(angle_deg) - 90) < ANGLE_THRESHOLD:
            # Línea vertical
            vertical_lines.append(((x1, y1, x2, y2), length))

# ----------------------------------------------------------------------------------
# !!! ELIMINACIÓN DE LA LÓGICA DÉBIL Y USO DE LA LÓGICA ROBUSTA (CEREBRO) !!!
# ----------------------------------------------------------------------------------

print("-" * 50)
print(
    f"Líneas H/V detectadas (antes de la lógica robusta): {len(horizontal_lines)}H, {len(vertical_lines)}V"
)
print(
    f"Líneas CANDIDATAS pasadas al filtro robusto: {len(horizontal_lines)}H, {len(vertical_lines)}V"
)

# Convertimos las listas a listas de tuplas (coords, length) para el filtro
horizontal_lines_raw = horizontal_lines
vertical_lines_raw = vertical_lines

# Paso Crítico: Filtrar las líneas que tocan la madera (zonas oscuras/rojas)
horizontal_lines = filter_by_color(original_image, horizontal_lines_raw)
vertical_lines = filter_by_color(original_image, vertical_lines_raw)

# ----------------------------------------------------------------------------------
# !!! USO DE LA LÓGICA ROBUSTA (Ahora solo con líneas candidatas) !!!
# ----------------------------------------------------------------------------------

print("-" * 50)
print(
    f"Líneas H/V detectadas (después del filtro de color): {len(horizontal_lines)}H, {len(vertical_lines)}V"
)

# Ejecución del algoritmo robusto (Ahora trabaja sobre un set de datos más limpios)
best_lines_coords, best_score, final_corners = find_best_quadrilateral(
    horizontal_lines, vertical_lines, original_image.shape
)

print("-" * 50)
print("ANÁLISIS DE ROBUSTEZ GEOMÉTRICA FINAL")
print(f"Mejor Puntuación Encontrada: {best_score:.2f}")

if best_lines_coords is None:
    print(
        "El filtro robusto NO ENCONTRÓ la mesa. Revise ANGLE_THRESHOLD y score_quadrilateral."
    )

if best_lines_coords is not None:

    # 1. Dibujar SÓLO las líneas que forman el mejor cuadrilátero
    print(
        "Las 4 líneas que definen la mesa (Mejor Cuadrilátero) han sido seleccionadas."
    )

    for i, line_coords in enumerate(best_lines_coords):
        x1, y1, x2, y2 = line_coords
        # Dibujamos las líneas seleccionadas en un color notorio
        cv2.line(
            image_with_filtered_lines, (x1, y1), (x2, y2), (0, 255, 255), 4
        )  # Amarillo Fuerte

    # 2. Dibujar las esquinas ORDENADAS (El Output que necesita la Homografía)
    print("\nCoordenadas de Origen (Esquinas) calculadas automáticamente:")
    print(f"Orden: TL, TR, BR, BL")
    print(final_corners)

    for corner in final_corners:
        cv2.circle(image_with_filtered_lines, corner, 10, (255, 0, 255), -1)  # Magenta

else:
    print(
        "Error: No se encontró ningún cuadrilátero válido con la proporción y área esperadas."
    )
    print("Intenta ajustar el ANGLE_THRESHOLD o los umbrales de Canny/Hough.")

# Muestra resultados forzando el redimensionamiento y posición (especialmente útil en Linux/Wayland)

# Crear ventanas con un nombre específico
cv2.namedWindow("Imagen Original", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Bordes (Canny)", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(
    "Mejor Cuadrilátero (Lineas y Esquinas)", cv2.WINDOW_NORMAL
)  # WINDOW_NORMAL permite redimensionar

# Si la imagen final es muy grande o muy pequeña, la redimensionamos para visualización
display_image = image_with_filtered_lines.copy()
"""
if display_image.shape[0] > 1000 or display_image.shape[1] > 1000:
    # Si es muy grande, la reducimos a la mitad para que quepa en pantalla
    display_image = cv2.resize(
        display_image, (display_image.shape[1] // 2, display_image.shape[0] // 2)
    )
"""

cv2.imshow("Imagen Original", original_image)
cv2.imshow("Bordes (Canny)", edges)
cv2.imshow("Mejor Cuadrilátero (Lineas y Esquinas)", display_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
