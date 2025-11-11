"""
Módulo de Integración Visual de Homografía con Pygame (Fase 5, Paso 6).

Este script actúa como el punto de integración y verificación final de la Fase 5.
Combina el detector de esquinas, el cálculo de Homografía y la corrección de
orientación para mostrar la mesa de billar perfectamente rectificada (vista cenital)
y siempre en un formato horizontal estándar (1000x500 píxeles).

Propósito principal:
1.  Validación del Pipeline: Confirmar que la detección de IA + matemáticas de
    perspectiva funcionan en conjunto.
2.  Visualización de Alto Nivel: Presentar la imagen final del plano de juego
    que serviría de fondo para la IA de juego (movimiento de bolas).
"""

import itertools
import math

import cv2
import numpy as np
import pygame

# =========================================================================
# I. FUNCIONES MATEMÁTICAS AUXILIARES (EL NÚCLEO DE ROBUSTEZ)
# =========================================================================


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
    # SELECCIÓN POR ÁREA MÍNIMA (EL MÁS INTERIOR)
    # --------------------------------------------------------
    if not all_candidates:
        return None, 0.0, None

    # 1. Filtro de Área Mínima
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


# --- FUNCIONES DE HOMOGRAFÍA Y PYGAME ---


def _calcular_orientacion_mesa(esquinas):
    # (Esta función auxiliar la tienes bien en tu script)
    # ...
    tl, tr, br, bl = esquinas
    ancho_top = np.linalg.norm(np.array(tr) - np.array(tl))
    ancho_bottom = np.linalg.norm(np.array(br) - np.array(bl))
    alto_left = np.linalg.norm(np.array(bl) - np.array(tl))
    alto_right = np.linalg.norm(np.array(br) - np.array(tr))
    ancho_promedio = (ancho_top + ancho_bottom) / 2
    alto_promedio = (alto_left + alto_right) / 2

    if ancho_promedio > alto_promedio:
        return "horizontal"
    else:
        return "vertical"


def calcular_matriz_homografia_orientada(
    esquinas_origen, ancho_final_h=1000, alto_final_h=500
):
    """
    Calcula H. Si la mesa es vertical, ajusta los puntos de destino
    para un lienzo vertical (ej. 500x1000) que luego será rotado.
    """

    # Llama a la función que ya tienes para saber la orientación
    orientacion = _calcular_orientacion_mesa(esquinas_origen)
    source_points = np.float32(esquinas_origen)

    if orientacion == "vertical":
        # La mesa original es vertical (más alta que ancha).
        # Mapeamos a un lienzo de destino vertical (ej. 500 de ancho x 1000 de alto)
        # para preservar la proporción.
        destination_points = np.float32(
            [
                [0, 0],
                [alto_final_h, 0],  # El destino ahora es 500 de ancho
                [alto_final_h, ancho_final_h],  # y 1000 de alto
                [0, ancho_final_h],
            ]
        )
    else:
        # La mesa original es horizontal. Mapeamos al lienzo estándar.
        destination_points = np.float32(
            [
                [0, 0],
                [ancho_final_h, 0],
                [ancho_final_h, alto_final_h],
                [0, alto_final_h],
            ]
        )

    # Calcular la matriz H
    H_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

    # Devolvemos la matriz Y la orientación, la necesitaremos
    return H_matrix, orientacion


def mostrar_imagen_pygame(imagen_cv, titulo="Mesa Aplanada (Pygame)"):
    """
    Muestra una imagen OpenCV (BGR) en una ventana de Pygame.
    """
    try:
        # Convertir de OpenCV (BGR) a Pygame (RGB)
        imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)

        # Convertir array de NumPy a superficie de Pygame (Swap X/Y axes)
        height, width, _ = imagen_rgb.shape
        pygame_surface = pygame.surfarray.make_surface(imagen_rgb.swapaxes(0, 1))

        # Inicializar Pygame
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(titulo)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.blit(pygame_surface, (0, 0))
            pygame.display.flip()

        pygame.quit()
    except Exception as e:
        print(f"Error al mostrar imagen en Pygame: {e}")
        pygame.quit()


def detectar_esquinas_mesa(imagen_cv):
    """
    Función principal. Toma una imagen y devuelve las 4 esquinas del área de juego
    ordenadas [TL, TR, BR, BL] o None si falla.
    """

    original_image = imagen_cv.copy()
    image_area = original_image.shape[0] * original_image.shape[1]

    # --- 1. Pre-procesamiento (Canny, Hough) ---
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 40, 125)  # Parámetros validados
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=30, minLineLength=150, maxLineGap=20
    )

    if lines is None:
        print("Error en Table_Detector: No se detectaron líneas iniciales con Hough.")
        return None

    # --- 2. Clasificación H/V y Filtros ---
    horizontal_lines = []
    vertical_lines = []

    # Bucle de clasificación (esto ya lo tienes)
    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)

        MIN_LINE_LENGTH = 30  # Usando el valor de tu última prueba
        ANGLE_THRESHOLD = 45  # Usando el valor de tu última prueba

        if length < MIN_LINE_LENGTH:
            continue
        if abs(angle_deg) < ANGLE_THRESHOLD or abs(angle_deg) > (180 - ANGLE_THRESHOLD):
            horizontal_lines.append(((x1, y1, x2, y2), length))
        elif abs(abs(angle_deg) - 90) < ANGLE_THRESHOLD:
            vertical_lines.append(((x1, y1, x2, y2), length))

    # (Aquí puedes añadir tu filtro de color si lo tenías)
    # horizontal_lines = filter_by_color(original_image, horizontal_lines)
    # vertical_lines = filter_by_color(original_image, vertical_lines)

    if not horizontal_lines or not vertical_lines:
        print("Error en Table_Detector: No hay suficientes líneas H/V candidatas.")
        return None

    # --- 3. Lógica de "Más Interior" (Tu heurística ganadora) ---
    best_lines_coords, best_score, final_corners = find_best_quadrilateral(
        horizontal_lines, vertical_lines, original_image.shape
    )

    if final_corners:
        print("Table_Detector: ¡Éxito! Esquinas encontradas.")
        return final_corners
    else:
        print(
            "Error en Table_Detector: La heurística de 'Más Interior' no encontró un cuadrilátero."
        )
        return None


# =========================================================================
# III. LÓGICA DE EJECUCIÓN PRINCIPAL (Prueba de P4 + Homografía + Pygame)
# =========================================================================

# --- 1. Definir Constantes de Salida ---
ANCHO_FINAL_HORIZONTAL = 1000
ALTO_FINAL_HORIZONTAL = 500

# --- 2. Cargar Imagen de Prueba ---
image_path = "detect_balls/tests/Black/test_pool_table_1.png"
image_path = "detect_balls/tests/Black/test_pool_table_2.png"
image_path = "detect_balls/tests/Black/test_pool_table_3.png"

imagen_cv_original = cv2.imread(image_path)

if imagen_cv_original is None:
    print(f"Error fatal: No se pudo cargar la imagen de prueba en {image_path}")
else:
    print(f"Imagen de prueba cargada: {image_path}")

    # --- 3. Detección de Mesa (P4) ---
    # ¡Ahora esta función SÍ existe!
    esquinas_detectadas = detectar_esquinas_mesa(imagen_cv_original)

    if esquinas_detectadas:
        print(f"Éxito de P4: Esquinas detectadas -> {esquinas_detectadas}")

        # --- 4. Cálculo de Homografía (Orientada) ---
        # (Asegúrate de tener esta función definida también)
        H_matrix, orientacion_original = calcular_matriz_homografia_orientada(
            esquinas_detectadas, ANCHO_FINAL_HORIZONTAL, ALTO_FINAL_HORIZONTAL
        )

        print(f"Orientación de mesa detectada: {orientacion_original}")

        # --- 5. Aplicar Homografía (Warp) ---
        if orientacion_original == "vertical":
            lienzo_ancho = ALTO_FINAL_HORIZONTAL  # 500
            lienzo_alto = ANCHO_FINAL_HORIZONTAL  # 1000
        else:
            lienzo_ancho = ANCHO_FINAL_HORIZONTAL  # 1000
            lienzo_alto = ALTO_FINAL_HORIZONTAL  # 500

        mesa_transformada = cv2.warpPerspective(
            imagen_cv_original, H_matrix, (lienzo_ancho, lienzo_alto)
        )

        # --- 6. Rotación Condicional ---
        if orientacion_original == "vertical":
            print("Rotando imagen vertical a formato horizontal...")
            mesa_transformada = cv2.rotate(
                mesa_transformada, cv2.ROTATE_90_COUNTERCLOCKWISE
            )

        # --- 7. Mostrar con Pygame ---
        print("Mostrando resultado final en Pygame (cierra la ventana para salir).")
        mostrar_imagen_pygame(
            mesa_transformada, titulo="Mesa Aplanada y Orientada (1000x500)"
        )

    else:
        print("Fallo en la prueba: table_detector no encontró esquinas.")
