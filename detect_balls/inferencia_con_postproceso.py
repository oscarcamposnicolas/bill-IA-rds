# inferencia_con_razonamiento_final.py
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

TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Classic2")
output_dir = os.path.join(
    base_project_dir, "tests", "results_razonamiento_final_Classic2"
)

CONFIDENCE_THRESHOLD = 0.4

# Lista de clases del modelo HÍBRIDO (25 clases)
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

# --- BASE DE CONOCIMIENTO PARA EL RAZONAMIENTO ---

DELATORES_BE = {
    "be_blue_10",
    "be_dred_15",
    "be_green_14",
    # "be_purple_13",
    # "be_purple_5",
    "be_pink_4",
    "be_pink_12",
    "be_red_11",
    "be_yellow_9",
}
DELATORES_CLASSIC = {
    "blue_10",
    "dred_15",
    "green_14",
    "orange_13",
    "orange_5",
    # "purple_12",
    # "purple_4",
    "red_11",
    "yellow_9",
}

# Tu mapa de corrección, la "inteligencia experta"
MAPA_CORRECCION_CONTEXTUAL = {
    # Si el contexto es CLASSIC pero el modelo predice una bola BE...
    "be_pink_4": "red_3",
    "be_pink_12": "red_11",
    "be_purple_5": "purple_4",
    "be_purple_13": "purple_12",
    # "be_yellow_9": "yellow_9",
    # "be_dred_15": "dred_15",
    # "be_blue_10": "blue_10",
    # "be_green_14": "green_14",
    # "be_red_11": "red_11",
    # Si el contexto es BLACK_EDITION pero el modelo predice una bola Classic...
    "purple_4": "be_purple_5",
    "purple_12": "be_purple_13",
    "yellow_9": "be_yellow_9",
}

# --- FIN DE LA CONFIGURACIÓN ---


def post_procesar_con_razonamiento(detecciones_yolo):
    puntuacion_be = sum(
        d["conf"] for d in detecciones_yolo if d["clase"] in DELATORES_BE
    )
    puntuacion_classic = sum(
        d["conf"] for d in detecciones_yolo if d["clase"] in DELATORES_CLASSIC
    )

    contexto = "black_edition" if puntuacion_be > puntuacion_classic else "classic"
    print(
        f"\n  > Puntuación 'classic': {puntuacion_classic:.2f} | Puntuación 'be': {puntuacion_be:.2f} -> Contexto Decidido: '{contexto}'"
    )

    detecciones_finales = []
    for deteccion in detecciones_yolo:
        etiqueta_yolo = deteccion["clase"]
        etiqueta_corregida = etiqueta_yolo

        # 1. Aplicar corrección contextual si es necesario
        if contexto == "classic" and etiqueta_yolo in DELATORES_BE:
            etiqueta_corregida = MAPA_CORRECCION_CONTEXTUAL.get(
                etiqueta_yolo, etiqueta_yolo
            )
            print(
                f"    - CORRECCIÓN (Contexto Classic): YOLO dijo '{etiqueta_yolo}', se corrige a '{etiqueta_corregida}'."
            )
        elif contexto == "black_edition" and etiqueta_yolo in DELATORES_CLASSIC:
            etiqueta_corregida = MAPA_CORRECCION_CONTEXTUAL.get(
                etiqueta_yolo, etiqueta_yolo
            )
            print(
                f"    - CORRECCIÓN (Contexto BE): YOLO dijo '{etiqueta_yolo}', se corrige a '{etiqueta_corregida}'."
            )

        # 2. Limpiar el prefijo 'be_' para la etiqueta final
        etiqueta_simple = etiqueta_corregida.replace("be_", "")

        deteccion["etiqueta_final"] = etiqueta_simple
        detecciones_finales.append(deteccion)

    return detecciones_finales


def main():
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
            continue

        detecciones_yolo = [
            {
                "clase": CLASES_HIBRIDAS[int(box.cls[0])],
                "conf": box.conf[0],
                "box_xyxy": box.xyxy[0],
            }
            for box in result.boxes
        ]

        detecciones_finales = post_procesar_con_razonamiento(detecciones_yolo)

        print("  > Resultados Finales para la Imagen:")
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Detecciones Corregidas en {image_filename}")

        for deteccion in detecciones_finales:
            x1, y1, x2, y2 = map(int, deteccion["box_xyxy"])
            etiqueta_final = deteccion["etiqueta_final"]
            confianza = deteccion["conf"]

            print(f"    - Bola: {etiqueta_final} (Confianza original: {confianza:.2f})")

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)
            text_y = y1 - 10 if y1 > 10 else y1 + 20
            ax.text(
                x1,
                text_y,
                f"{etiqueta_final} {confianza:.2f}",
                color="black",
                fontsize=10,
                bbox=dict(facecolor="cyan", alpha=0.8),
            )

        plt.axis("off")
        output_image_path = os.path.join(output_dir, f"resultado_{image_filename}")
        plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print("\n\n--- Proceso de inferencia final completado. ---")


if __name__ == "__main__":
    main()
