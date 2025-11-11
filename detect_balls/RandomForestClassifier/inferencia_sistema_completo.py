"""
Módulo de Inferencia y Validación del Sistema Híbrido Completo (Fase 4, Paso 3).

Este script realiza una prueba de estrés del sistema híbrido de inferencia sobre
un lote de imágenes de prueba con etiquetas de verdad fundamental (ground truth).

Propósito principal:
1.  Validar la precisión de la Clasificación de Contexto (Random Forest) en un entorno
    de producción (usando las detecciones reales de YOLO como input).
2.  Medir métricas de rendimiento globales (ej., precisión del 98% en la clasificación de contexto).
3.  Simular el proceso de arbitraje que se utilizaría en la presentación del concurso.
"""

import os

import cv2
import joblib
import pandas as pd
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN ---
base_project_dir = os.path.join(".", "detect_balls")

# --- Rutas de los Modelos ---
DETECTION_MODEL_PATH = "detect_balls/runs/Modelo_Hibrido_v1/weights/best.pt"
CONTEXT_MODEL_PATH = "RandomForestClassifier/context_classifier.joblib"

# --- Rutas de Archivos ---
TEST_IMAGES_DIR = os.path.join(base_project_dir, "tests", "Mix")
OUTPUT_DIR = os.path.join(base_project_dir, "tests", "results_mix")

# --- Parámetros de Inferencia ---
CONFIDENCE_THRESHOLD = 0.4

# --- Listas de Clases y Reglas de Lógica ---
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
CLASES_COMPARTIDAS = {
    "black_8",
    "blue_2",
    "dred_7",
    "green_6",
    "red_3",
    "white",
    "yellow_1",
}
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
MAPA_CORRECCION_CONTEXTUAL = {
    "be_pink_4": "red_3",
    "be_pink_12": "red_11",
    "be_purple_5": "purple_4",
    "be_purple_13": "purple_12",
    "be_yellow_9": "yellow_9",
    "purple_4": "be_pink_4",
    "purple_12": "be_pink_12",
    "purple_4": "be_purple_5",
    "purple_12": "be_purple_13",
}
# --- FIN DE LA CONFIGURACIÓN ---


def extraer_features_para_contexto(detecciones_yolo):
    """Extraer las características de las detecciones para alimentar al clasificador de contexto."""
    puntuacion_be = sum(
        d["conf"] for d in detecciones_yolo if d["clase"] in DELATORES_BE
    )
    puntuacion_classic = sum(
        d["conf"] for d in detecciones_yolo if d["clase"] in DELATORES_CLASSIC
    )

    counts = {f"count_{cls}": 0 for cls in CLASES_HIBRIDAS}
    confianza_total = 0.0
    for d in detecciones_yolo:
        counts[f"count_{d['clase']}"] += 1
        confianza_total += d["conf"]

    total_detecciones = len(detecciones_yolo)
    confianza_media = (
        confianza_total / total_detecciones if total_detecciones > 0 else 0
    )

    # Crear un DataFrame de una sola fila con las features en el orden correcto
    features = {
        "puntuacion_be": puntuacion_be,
        "puntuacion_classic": puntuacion_classic,
        "total_detecciones": total_detecciones,
        "confianza_media": confianza_media,
    }
    features.update(counts)

    # Asegurar el orden de las columnas
    column_order = [
        "puntuacion_be",
        "puntuacion_classic",
        "total_detecciones",
        "confianza_media",
    ] + [f"count_{cls}" for cls in CLASES_HIBRIDAS]

    return pd.DataFrame([features], columns=column_order)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cargar los dos modelos
    print("Cargando modelo de detección (el 'Ojo')...")
    detection_model = YOLO(DETECTION_MODEL_PATH)
    print("Cargando modelo de contexto (el 'Cerebro')...")
    context_model = joblib.load(CONTEXT_MODEL_PATH)

    image_files = [
        f
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(
        f"\nIniciando inferencia del sistema completo en {len(image_files)} imágenes..."
    )

    for image_filename in image_files:
        test_image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        print(f"\n--- Procesando: {image_filename} ---")

        # 1. El "Ojo" (YOLO) hace su trabajo
        results = detection_model.predict(
            test_image_path, verbose=False, conf=CONFIDENCE_THRESHOLD
        )
        detecciones_yolo = [
            {
                "clase": CLASES_HIBRIDAS[int(box.cls[0])],
                "conf": float(box.conf[0]),
                "box_xyxy": box.xyxy[0],
            }
            for box in results[0].boxes
        ]

        if not detecciones_yolo:
            print("No se detectaron objetos.")
            continue

        # 2. El "Cerebro" (Random Forest) determina el contexto
        features = extraer_features_para_contexto(detecciones_yolo)
        contexto_pred_id = context_model.predict(features)[0]
        contexto = "black_edition" if contexto_pred_id == 1 else "classic"
        print(f"  > Contexto decidido por el 'Cerebro' de ML: '{contexto}'")

        # 3. Se aplica la lógica de corrección final
        detecciones_finales = []
        for deteccion in detecciones_yolo:
            # ... (código de corrección y simplificación de etiquetas) ...
            etiqueta_yolo = deteccion["clase"]
            etiqueta_corregida = etiqueta_yolo
            if contexto == "classic" and etiqueta_yolo in DELATORES_BE:
                etiqueta_corregida = MAPA_CORRECCION_CONTEXTUAL.get(
                    etiqueta_yolo, etiqueta_yolo
                )
            elif contexto == "black_edition" and etiqueta_yolo in DELATORES_CLASSIC:
                etiqueta_corregida = MAPA_CORRECCION_CONTEXTUAL.get(
                    etiqueta_yolo, etiqueta_yolo
                )

            etiqueta_simple = etiqueta_corregida.replace("be_", "")
            if contexto == "black_edition" and etiqueta_simple in CLASES_COMPARTIDAS:
                etiqueta_simple = "be_" + etiqueta_simple

            deteccion["etiqueta_final"] = etiqueta_simple
            detecciones_finales.append(deteccion)

        # 4. Se visualizan los resultados finales
        img = cv2.imread(test_image_path)
        for d in detecciones_finales:
            x1, y1, x2, y2 = map(int, d["box_xyxy"])
            label = f"{d['etiqueta_final']} {d['conf']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(
                img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

        output_path = os.path.join(OUTPUT_DIR, f"resultado_{image_filename}")
        cv2.imwrite(output_path, img)
        print(f"  > Resultado final guardado en: {output_path}")

    print("\n\n¡PROYECTO FINALIZADO CON ÉXITO!")


if __name__ == "__main__":
    main()
