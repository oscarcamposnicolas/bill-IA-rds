"""
Módulo del Motor de Inferencia Híbrido (Fase 4).

Este script es el corazón del sistema de IA, combinando Deep Learning (YOLO)
y Machine Learning Clásico (Random Forest) para realizar un análisis completo
de la escena.

Propósito principal:
1.  **Detección (DL):** Utiliza YOLOv11 ('pool_hybrid.pt') para detectar y
    localizar todas las bolas en la imagen.
2.  **Extracción de Features (CV):** Procesa las detecciones de YOLO para
    construir un vector de características numéricas (conteo de bolas).
3.  **Clasificación (ML):** Utiliza un Random Forest ('context_classifier.joblib')
    para clasificar el contexto de la mesa (ej. 'Classic' o 'BE').
4.  **Output Dual:** Devuelve tanto los centroides de las bolas (para la Homografía)
    como la etiqueta del contexto (para la lógica de reglas).
"""

import os

import cv2
import joblib
import pandas as pd
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN Y CARGA DE MODELOS ---
print("Iniciando motor de inferencia...")
DETECTION_MODEL_PATH = "pool_hybrid.pt"
CONTEXT_MODEL_PATH = "context_classifier.joblib"

# Listas de clases y reglas
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
    "purple_4": "be_purple_5",
    "purple_12": "be_purple_13",
}

# Cargar los dos modelos
try:
    detection_model = YOLO(DETECTION_MODEL_PATH)
    context_model = joblib.load(CONTEXT_MODEL_PATH)
    print("Motor de IA: Modelos de detección y contexto cargados correctamente.")
except Exception as e:
    print(f"Error fatal al cargar modelos en el motor: {e}")
    detection_model = None
    context_model = None

# --- 2. FUNCIONES DE LÓGICA ---


def extraer_features_para_contexto(detecciones_yolo):
    """Prepara los datos de entrada para el clasificador de contexto (Random Forest)."""
    puntuacion_be = sum(
        d["conf"] for d in detecciones_yolo if d["clase"] in DELATORES_BE
    )
    puntuacion_classic = sum(
        d["conf"] for d in detecciones_yolo if d["clase"] in DELATORES_CLASSIC
    )

    counts = {f"count_{cls}": 0 for cls in CLASES_HIBRIDAS}
    confianza_total = sum(d["conf"] for d in detecciones_yolo)

    for d in detecciones_yolo:
        counts[f"count_{d['clase']}"] += 1

    total_detecciones = len(detecciones_yolo)
    confianza_media = (
        confianza_total / total_detecciones if total_detecciones > 0 else 0
    )

    features = {
        "puntuacion_be": puntuacion_be,
        "puntuacion_classic": puntuacion_classic,
        "total_detecciones": total_detecciones,
        "confianza_media": confianza_media,
    }
    features.update(counts)

    # Asegurar el orden de las columnas para que coincida con el del entrenamiento
    column_order = [
        "puntuacion_be",
        "puntuacion_classic",
        "total_detecciones",
        "confianza_media",
    ] + [f"count_{cls}" for cls in CLASES_HIBRIDAS]

    return pd.DataFrame([features], columns=column_order)


def refinar_etiquetas(detecciones_yolo, contexto):
    """
    Aplica corrección contextual y luego simplifica TODAS las etiquetas
    para que no contengan el prefijo 'be_'.
    """
    detecciones_refinadas = []
    for deteccion in detecciones_yolo:
        etiqueta_yolo = deteccion["clase"]
        etiqueta_corregida = etiqueta_yolo

        # 1. Aplicar corrección contextual si es necesario
        if contexto == "classic" and etiqueta_yolo in DELATORES_BE:
            etiqueta_corregida = MAPA_CORRECCION_CONTEXTUAL.get(
                etiqueta_yolo, etiqueta_yolo
            )
            # print(f"  - CORRECCIÓN (Classic): YOLO dijo '{etiqueta_yolo}', se corrige a '{etiqueta_corregida}'.")
        elif contexto == "black_edition" and etiqueta_yolo in DELATORES_CLASSIC:
            etiqueta_corregida = MAPA_CORRECCION_CONTEXTUAL.get(
                etiqueta_yolo, etiqueta_yolo
            )
            # print(f"  - CORRECCIÓN (BE): YOLO dijo '{etiqueta_yolo}', se corrige a '{etiqueta_corregida}'.")

        # 2. Simplificar la etiqueta final, eliminando siempre el prefijo 'be_'
        etiqueta_final_simple = etiqueta_corregida.replace("be_", "")

        deteccion["etiqueta_final"] = etiqueta_final_simple
        detecciones_refinadas.append(deteccion)

    return detecciones_refinadas


# --- 3. FUNCIÓN PRINCIPAL DE INFERENCIA ---
def realizar_inferencia(ruta_imagen_entrada, ruta_imagen_salida):
    print(ruta_imagen_entrada)
    print(ruta_imagen_salida)

    if not detection_model or not context_model:
        raise Exception("Los modelos de IA no están cargados.")

    # 1. El "Ojo" (YOLO) detecta las bolas
    print("1. El Ojo (YOLO) detecta las bolas")
    results = detection_model.predict(ruta_imagen_entrada, verbose=False, conf=0.4)
    detecciones_yolo = [
        {
            "clase": CLASES_HIBRIDAS[int(box.cls[0])],
            "conf": float(box.conf[0]),
            "box_xyxy": box.xyxy[0].tolist(),
        }
        for box in results[0].boxes
    ]

    if not detecciones_yolo:
        img = cv2.imread(ruta_imagen_entrada)
        cv2.imwrite(ruta_imagen_salida, img)
        return [], "No se detectaron bolas"

    # 2. El "Cerebro" (Random Forest) determina el contexto
    print("2. El Cerebro (Random Forest) determina el contexto")
    features = extraer_features_para_contexto(detecciones_yolo)
    contexto_pred_id = context_model.predict(features)[0]
    contexto = "black_edition" if contexto_pred_id == 1 else "classic"

    # 3. La lógica final refina las etiquetas
    print("3. La lógica final refina las etiquetas")
    detecciones_finales = refinar_etiquetas(detecciones_yolo, contexto)

    # 4. Dibujar resultados y guardar
    img = cv2.imread(ruta_imagen_entrada)
    for d in detecciones_finales:
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        label = f"{d['etiqueta_final']} {d['conf']:.2f}"
        color = (0, 255, 0)  # Verde en BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    cv2.imwrite(ruta_imagen_salida, img)

    return detecciones_finales, contexto
