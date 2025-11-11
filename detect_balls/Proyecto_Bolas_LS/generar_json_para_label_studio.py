"""
Módulo de Pre-procesamiento de Datos para Label Studio (Fase 3, Paso 3).

Este script genera un archivo JSON en el formato de tareas esperado por la API de
Label Studio. Este JSON es el input masivo que permite importar un lote completo
de imágenes para el etiquetado de forma programática.

Propósito principal:
1.  Formatear listas de URLs de imágenes (locales o en la nube) en la estructura de tareas.
2.  Facilitar la automatización de la adquisición de datos (Human-in-the-Loop).
3.  Establecer la base para el pre-etiquetado.
"""

import json
import os
import uuid
from urllib.parse import quote

# --- 1. CONFIGURACIÓN ---

# La ruta a la carpeta que contiene los archivos .txt de pre-etiquetado
ETIQUETAS_DIR = "/home/oscar/Documentos/Estudios/Curso.Especialista.IA/Proyecto/src/detect_balls/datasets/black_edition_raw/labels/"

# La ruta RELATIVA desde "DOCUMENT_ROOT" de Label Studio hasta las imágenes
# DOCUMENT_ROOT es: .../Proyecto/
# Las imágenes están en: .../Proyecto/src/Proyecto_Bolas_LS/imagenes/
RUTA_RELATIVA_IMAGENES = "src/Proyecto_Bolas_LS/imagenes"

# El nombre del archivo JSON de salida
ARCHIVO_SALIDA_JSON = "Proyecto_Bolas_LS/tasks.json"

# Lista de clases ORIGINAL para interpretar los IDs del modelo antiguo
CLASSES_ORIGINAL = [
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
]
# --- FIN DE LA CONFIGURACIÓN ---


def crear_tareas_finales():
    tasks = []
    print(f"Leyendo etiquetas desde: {ETIQUETAS_DIR}")

    for label_file in os.listdir(ETIQUETAS_DIR):
        if not label_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(label_file)[0]
        # Asumimos que la imagen tiene extensión .png, .jpg, o .jpeg
        # Esto es para encontrar el nombre completo del archivo de imagen
        nombre_imagen_completo = None
        # Recreamos el nombre original buscando la extensión
        # NOTA: Esto asume que estan las imágenes originales en alguna parte para saber su extensión
        # Si todas son png, por ejemplo, se puede simplificar.
        for ext in [".png", ".jpg", ".jpeg"]:
            # Solo para obtener el nombre correcto, no para acceder al archivo
            if os.path.exists(
                os.path.join(ETIQUETAS_DIR.replace("labels", "images"), base_name + ext)
            ):
                nombre_imagen_completo = base_name + ext
                break

        if not nombre_imagen_completo:
            print(f"No se pudo inferir la extensión para {base_name}. Saltando.")
            continue

        # Codificar el nombre del archivo para la URL (maneja espacios y caracteres especiales)
        nombre_imagen_url_encoded = quote(nombre_imagen_completo)

        # Construir la URL interna de Label Studio
        ruta_en_servidor = f"/data/local-files/?d={os.path.join(RUTA_RELATIVA_IMAGENES, nombre_imagen_url_encoded)}"

        task = {
            "data": {"image": ruta_en_servidor},
            "predictions": [{"model_version": "yolov8_pretrained", "result": []}],
        }

        with open(os.path.join(ETIQUETAS_DIR, label_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x, y, w, h = map(
                    float, [parts[0], parts[1], parts[2], parts[3], parts[4]]
                )

                task["predictions"][0]["result"].append(
                    {
                        "id": str(uuid.uuid4()),
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x - w / 2) * 100,
                            "y": (y - h / 2) * 100,
                            "width": w * 100,
                            "height": h * 100,
                            "rotation": 0,
                            "rectanglelabels": [CLASSES_ORIGINAL[int(class_id)]],
                        },
                    }
                )
        tasks.append(task)

    with open(ARCHIVO_SALIDA_JSON, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"\n¡Éxito! Se ha generado '{ARCHIVO_SALIDA_JSON}' con {len(tasks)} tareas.")


if __name__ == "__main__":
    crear_tareas_finales()
