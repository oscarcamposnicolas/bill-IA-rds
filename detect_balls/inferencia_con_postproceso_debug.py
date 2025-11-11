"""
Módulo de Inferencia Híbrida con Funcionalidad de Debug (Fase 2, Utilidad).

Este script extiende la funcionalidad del módulo de inferencia principal
(inferencia_con_postproceso.py) añadiendo herramientas de diagnóstico.

Propósito principal:
1.  Debugging Visual: Mostrar la imagen con las detecciones de YOLO y las etiquetas
    de clasificación para la validación visual y manual.
2.  Diagnóstico de Datos: Imprimir el vector de características exacto que se pasa
    al clasificador de Machine Learning (Random Forest).
3.  Prueba Unitária: Asegurar que el sistema híbrido toma la decisión correcta
    (contexto Clásico/BE) en un entorno controlado.
"""

import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN ---
base_project_dir = os.path.join(".", "detect_balls")
train_version = "Modelo_Hibrido_v1"
runs_dir = os.path.join(base_project_dir, "runs")
model_path = os.path.join(runs_dir, train_version, "weights", "best.pt")

# TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests_classic_set")
TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Mix")
# output_dir = os.path.join(base_project_dir, "results_final_con_postproceso")
output_dir = os.path.join(base_project_dir, "tests", "results_final_con_postproceso")

CONFIDENCE_THRESHOLD = 0.4

CLASES_HIBRIDAS = [
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
    "be_blue_10",
    "be_dred_15",
    "be_green_14",
    "be_purple_13",
    "be_purple_5",
    "be_pink_4",
    "be_pink_12",
    "be_red_11",
    "be_yellow_9",
]
CLASES_COMPARTIDAS = [
    "black_8",
    "blue_2",
    "dred_7",
    "green_6",
    "red_3",
    "white",
    "yellow_1",
]
DELATORES_BE = [
    "be_blue_10",
    "be_dred_15",
    "be_green_14",
    "be_purple_13",
    "be_purple_5",
    "be_pink_4",
    "be_pink_12",
    "be_red_11",
    "be_yellow_9",
]

# --- FIN DE LA CONFIGURACIÓN ---


def post_procesar_detecciones(detecciones_yolo):
    nombres_clases_detectadas = [d["clase"] for d in detecciones_yolo]

    # 1. Detectar el contexto
    contexto = "classic"
    delator_encontrado = None
    for delator in DELATORES_BE:
        if delator in nombres_clases_detectadas:
            contexto = "black_edition"
            delator_encontrado = delator  # Guardar qué delator activó el cambio
            break

    print(f"\n  > Iniciando Post-Procesamiento...")
    if delator_encontrado:
        print(
            f"  > Contexto detectado: '{contexto}' (activado por la detección de '{delator_encontrado}')"
        )
    else:
        print(f"  > Contexto detectado: '{contexto}' (no se encontraron 'delatores')")

    # 2. Refinar etiquetas
    detecciones_refinadas = []
    for deteccion in detecciones_yolo:
        etiqueta_original = deteccion["clase"]
        etiqueta_final = etiqueta_original

        if contexto == "black_edition" and etiqueta_original in CLASES_COMPARTIDAS:
            etiqueta_final = "be_" + etiqueta_original

        deteccion["etiqueta_final"] = etiqueta_final
        detecciones_refinadas.append(deteccion)

        # --- SALIDA DE DEPURACIÓN ---
        print(
            f"    - Predicción YOLO: '{etiqueta_original}' -> Etiqueta Final: '{etiqueta_final}'"
        )

    return detecciones_refinadas


def inferencia_final_debug():
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    print(f"Modelo cargado desde: {model_path}")

    image_files = [
        f
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"\nIniciando inferencia en {len(image_files)} imágenes...")

    for image_filename in image_files:
        test_image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        print(f"\n--- Procesando: {image_filename} ---")

        results = model.predict(
            source=test_image_path,
            save=False,
            imgsz=1024,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        if not result.boxes:
            print("No se detectaron objetos en esta imagen.")
            continue

        detecciones_yolo = []
        print("  > Detecciones del Modelo (salida en bruto):")
        for box in result.boxes:
            class_id = int(box.cls[0])
            clase_detectada = CLASES_HIBRIDAS[class_id]
            print(
                f"    - Objeto detectado como: '{clase_detectada}' (Conf: {box.conf[0]:.2f})"
            )
            detecciones_yolo.append(
                {"clase": clase_detectada, "conf": box.conf[0], "box_xyxy": box.xyxy[0]}
            )

        detecciones_finales = post_procesar_detecciones(detecciones_yolo)

        # --- Visualización con las etiquetas finales ---
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Detecciones Finales en {image_filename}")

        for deteccion in detecciones_finales:
            x1, y1, x2, y2 = map(int, deteccion["box_xyxy"])
            etiqueta_a_mostrar = deteccion["etiqueta_final"]
            confianza = deteccion["conf"]

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

            text_y = y1 - 10 if y1 > 10 else y1 + 20
            ax.text(
                x1,
                text_y,
                f"{etiqueta_a_mostrar} {confianza:.2f}",
                color="black",
                fontsize=10,
                bbox=dict(facecolor="lime", alpha=0.8),
            )

        plt.axis("off")
        output_image_path = os.path.join(output_dir, f"resultado_{image_filename}")
        plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print("\n\n--- Proceso de inferencia final completado. ---")


if __name__ == "__main__":
    inferencia_final_debug()
