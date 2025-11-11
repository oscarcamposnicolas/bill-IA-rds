# convertir_ls_a_yolo_FINAL.py
import json
import os
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import unquote

# --- 1. CONFIGURACIÓN ---

# Ruta al archivo JSON exportado desde Label Studio
LABEL_STUDIO_JSON_PATH = "Proyecto_Bolas_LS/ls-export.json"

# Ruta al archivo XML que guardaste desde la interfaz de Label Studio
LABEL_STUDIO_XML_CONFIG_PATH = "Proyecto_Bolas_LS/config.xml"

# Carpeta donde se guardarán las nuevas etiquetas en formato YOLO (.txt)
YOLO_LABELS_OUTPUT_DIR = "Proyecto_Bolas_LS/final_yolo_labels/"

# --- FIN DE LA CONFIGURACIÓN ---


def leer_clases_desde_xml(xml_path):
    """Lee el archivo de configuración XML de Label Studio para extraer las clases en orden."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        clases = [label.get("value") for label in root.findall(".//Label")]
        print(f"Clases leídas desde '{xml_path}': {len(clases)} clases encontradas.")
        return clases
    except Exception as e:
        print(f"Error al leer o parsear el archivo XML '{xml_path}': {e}")
        return None


def convertir_anotaciones_final():
    # 1. Construir la lista de clases desde el XML (Fuente Única de Verdad)
    clases_maestra = leer_clases_desde_xml(LABEL_STUDIO_XML_CONFIG_PATH)
    if not clases_maestra:
        return
    class_to_id = {name: i for i, name in enumerate(clases_maestra)}

    # 2. Cargar el archivo JSON de Label Studio
    os.makedirs(YOLO_LABELS_OUTPUT_DIR, exist_ok=True)
    try:
        with open(LABEL_STUDIO_JSON_PATH, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{LABEL_STUDIO_JSON_PATH}'.")
        return

    print(f"Se han cargado {len(data)} tareas desde Label Studio.")

    # 3. Procesar cada tarea y convertir a formato YOLO
    for task in data:
        try:
            image_path = task["image"]
            image_filename = os.path.basename(unquote(image_path))
            parts = image_filename.split("-", 1)
            if len(parts) > 1 and (len(parts[0]) == 8 or len(parts[0]) > 20):
                image_filename = parts[1]
            yolo_txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        except KeyError:
            print(
                f"Advertencia: La tarea con ID {task.get('id')} no tiene clave 'image'. Saltando."
            )
            continue

        output_path = os.path.join(YOLO_LABELS_OUTPUT_DIR, yolo_txt_filename)
        with open(output_path, "w") as f_yolo:
            if "label" in task and task["label"]:
                for annotation in task["label"]:
                    try:
                        class_name = annotation["rectanglelabels"][0]
                        if class_name in class_to_id:
                            class_id = class_to_id[class_name]
                            x_ls, y_ls = annotation["x"], annotation["y"]
                            w_ls, h_ls = annotation["width"], annotation["height"]

                            x_norm, y_norm, w_norm, h_norm = (
                                x_ls / 100.0,
                                y_ls / 100.0,
                                w_ls / 100.0,
                                h_ls / 100.0,
                            )
                            x_center, y_center = x_norm + (w_norm / 2), y_norm + (
                                h_norm / 2
                            )

                            f_yolo.write(
                                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                            )
                        else:
                            print(
                                f"ADVERTENCIA: La clase '{class_name}' de '{image_filename}' no se encontró en 'config.xml'."
                            )
                    except KeyError as e:
                        print(
                            f"Advertencia: Datos incompletos en anotación para '{image_filename}'. Error: {e}."
                        )

    print(f"\n¡Conversión completada! Revisa la carpeta '{YOLO_LABELS_OUTPUT_DIR}'.")


if __name__ == "__main__":
    convertir_anotaciones_final()
