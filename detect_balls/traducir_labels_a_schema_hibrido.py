"""
Módulo de Traducción de Esquema de Clases (Fase 2, Utilidad).

Este script realiza una traducción crítica de las etiquetas de los datasets originales
(que pueden tener un alto número de clases específicas) a un esquema de clases más
reducido y optimizado. El objetivo es consolidar clases similares o redundantes
para mejorar la generalización del modelo de detección YOLO (P1).

Propósito principal:
1.  Reducir la complejidad del problema de clasificación.
2.  Mejorar el rendimiento del modelo en tareas de detección con un dataset limitado.
3.  Asegurar la compatibilidad con el esquema final de clases híbrido (ej., Black Edition).
"""

import os
import shutil

# --- CONFIGURACIÓN ---

# Directorio con las etiquetas del dataset unificado (el de 32 clases)
LABELS_ORIGINALES_DIR = "detect_balls/dataset_unificado_aumentado/labels/"
# Directorio de salida para las nuevas etiquetas "híbridas"
LABELS_HIBRIDAS_DIR = "detect_balls/dataset_hibrido/labels/"

# Lista MAESTRA ORIGINAL con la que se generaron las etiquetas (32 clases)
CLASES_MAESTRA_ORIGINAL = [
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
    "be_black_8",
    "be_blue_10",
    "be_blue_2",
    "be_dred_15",
    "be_dred_7",
    "be_green_14",
    "be_green_6",
    "be_purple_13",
    "be_purple_5",
    "be_pink_4",
    "be_pink_12",
    "be_red_11",
    "be_red_3",
    "be_white",
    "be_yellow_1",
    "be_yellow_9",
]

# NUEVA lista HÍBRIDA de 25 clases
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
# --- FIN DE LA CONFIGURACIÓN ---


def traducir_etiquetas_hibridas():
    id_a_nombre_original = {i: name for i, name in enumerate(CLASES_MAESTRA_ORIGINAL)}
    nombre_a_id_hibrido = {name: i for i, name in enumerate(CLASES_HIBRIDAS)}

    # Mapeo de lógica para unificar clases
    # La clave es el nombre original, el valor es el nombre en el nuevo esquema
    mapa_traduccion = {
        # Bolas que son únicas en el set BE
        "be_blue_10": "be_blue_10",
        "be_dred_15": "be_dred_15",
        "be_green_14": "be_green_14",
        "be_purple_13": "be_purple_13",
        "be_purple_5": "be_purple_5",
        "be_pink_4": "be_pink_4",
        "be_pink_12": "be_pink_12",
        "be_red_11": "be_red_11",
        "be_yellow_9": "be_yellow_9",
        # Bolas que son únicas en el set Clásico (las rayadas blancas)
        "blue_10": "blue_10",
        "dred_15": "dred_15",
        "green_14": "green_14",
        "orange_13": "orange_13",
        "purple_12": "purple_12",
        "red_11": "red_11",
        "yellow_9": "yellow_9",
        # Bolas sólidas que se unifican (el prefijo 'be_' desaparece)
        "black_8": "black_8",
        "be_black_8": "black_8",
        "blue_2": "blue_2",
        "be_blue_2": "blue_2",
        "dred_7": "dred_7",
        "be_dred_7": "dred_7",
        "green_6": "green_6",
        "be_green_6": "green_6",
        "orange_5": "orange_5",  # No hay 'be_orange_5', pero lo mantenemos por coherencia
        "purple_4": "purple_4",  # No hay 'be_purple_4' (es rosa), pero lo mantenemos
        "red_3": "red_3",
        "be_red_3": "red_3",
        "white": "white",
        "be_white": "white",
        "yellow_1": "yellow_1",
        "be_yellow_1": "yellow_1",
    }

    for split in ["train", "valid", "test"]:
        print(f"Traduciendo etiquetas de: {split}...")
        dir_origen = os.path.join(LABELS_ORIGINALES_DIR, split)
        dir_destino = os.path.join(LABELS_HIBRIDAS_DIR, split)

        for filename in os.listdir(dir_origen):
            with open(os.path.join(dir_origen, filename), "r") as f_in, open(
                os.path.join(dir_destino, filename), "w"
            ) as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    # id_original = int(parts[0])
                    id_original = int(float(parts[0]))
                    coords = " ".join(parts[1:])

                    nombre_original = id_a_nombre_original[id_original]
                    nombre_hibrido = mapa_traduccion.get(nombre_original)

                    if nombre_hibrido and nombre_hibrido in nombre_a_id_hibrido:
                        id_hibrido = nombre_a_id_hibrido[nombre_hibrido]
                        f_out.write(f"{id_hibrido} {coords}\n")
                    else:
                        print(
                            f"ADVERTENCIA: No se pudo traducir la clase '{nombre_original}'"
                        )

    print("\n¡Traducción de etiquetas al esquema híbrido completada!")


if __name__ == "__main__":
    traducir_etiquetas_hibridas()
