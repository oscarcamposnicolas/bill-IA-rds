# aumentar_dataset_be.py
import os
import random

import albumentations as A
import cv2
import numpy as np

# --- CONFIGURACIÓN ---
# Directorios de origen (tus 60 imágenes y etiquetas corregidas)
IMAGENES_ORIGINALES_DIR = "detect_pockets/images/"
LABELS_ORIGINALES_DIR = "detect_pockets/final_yolo_labels/"

# Directorios de salida para los datos aumentados
IMAGENES_AUMENTADAS_DIR = "detect_pockets/dataset_aumentado/images/"
LABELS_AUMENTADAS_DIR = "detect_pockets/dataset_aumentado/labels/"

# Número de versiones nuevas que quieres crear por cada imagen original
NUM_AUMENTACIONES_POR_IMAGEN = 15
# --- FIN DE LA CONFIGURACIÓN ---


def augment_dataset():
    # Crear directorios de salida
    os.makedirs(IMAGENES_AUMENTADAS_DIR, exist_ok=True)
    os.makedirs(LABELS_AUMENTADAS_DIR, exist_ok=True)

    # Definir el pipeline de aumentación. ¡Aquí está la magia!
    # BboxParams se asegura de que las cajas se transformen junto con la imagen.
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3,
                border_mode=cv2.BORDER_CONSTANT,
            ),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    image_files = [
        f
        for f in os.listdir(IMAGENES_ORIGINALES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(
        f"Iniciando aumentación para {len(image_files)} imágenes. Se crearán {NUM_AUMENTACIONES_POR_IMAGEN} versiones de cada una."
    )

    for image_name in image_files:
        image_path = os.path.join(IMAGENES_ORIGINALES_DIR, image_name)
        base_name = os.path.splitext(image_name)[0]
        label_base_name = base_name.replace(" ", "_")
        label_path = os.path.join(LABELS_ORIGINALES_DIR, label_base_name + ".txt")

        if not os.path.exists(label_path):
            print(
                "-------------------------------------------------------------------------"
            )
            print(f"label_path: {label_path}")
            print(f"image_path: {image_path}")
            print(f"base_name: {base_name}")
            print(f"image_name: {image_name}")
            continue

        # Cargar imagen y etiquetas
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []

        """
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id, x, y, w, h = map(float, parts)
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, 1.0)
                w = np.clip(w, 0.0, 1.0)
                h = np.clip(h, 0.0, 1.0)
                bboxes.append([x, y, w, h])
                class_labels.append(int(class_id))
        """

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Advertencia: Línea malformada en {label_path}, saltando.")
                    continue

                # 1. Lee los 5 valores
                class_id = int(float(parts[0]))
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # --- INICIO DE LA CORRECCIÓN ---
                # 2. Convierte a formato [x_min, y_min, x_max, y_max]
                x_min = x_center_norm - (width_norm / 2)
                y_min = y_center_norm - (
                    height_norm / 2
                )  # <-- Aquí se produce el -4.99e-07
                x_max = x_center_norm + (width_norm / 2)
                y_max = y_center_norm + (height_norm / 2)

                # 3. Forzamos (clip) los valores MIN/MAX al rango [0.0, 1.0]
                x_min = np.clip(x_min, 0.0, 1.0)
                y_min = np.clip(y_min, 0.0, 1.0)  # <-- Esto convierte -4.99e-07 en 0.0
                x_max = np.clip(x_max, 0.0, 1.0)
                y_max = np.clip(y_max, 0.0, 1.0)

                # 4. Evitar bboxes de área cero (que pueden ocurrir después del clip)
                if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
                    print(f"Advertencia: Bbox nula en {label_path}. Saltando bbox.")
                    continue

                # 5. Volver a convertir a formato YOLO [x_center, y_center, w, h]
                #    porque esto es lo que 'transform' espera (format='yolo' en línea 38)
                x_clean = (x_min + x_max) / 2
                y_clean = (y_min + y_max) / 2
                w_clean = x_max - x_min
                h_clean = y_max - y_min
                # --- FIN DE LA CORRECCIÓN ---

                # 6. Añadir los datos "limpios"
                bboxes.append([x_clean, y_clean, w_clean, h_clean])
                class_labels.append(class_id)

        # Generar N versiones aumentadas de la imagen
        for i in range(NUM_AUMENTACIONES_POR_IMAGEN):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]

            # Guardar la nueva imagen
            aug_image_name = f"{base_name}_aug_{i}.jpg"
            cv2.imwrite(
                os.path.join(IMAGENES_AUMENTADAS_DIR, aug_image_name),
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
            )

            # Guardar las nuevas etiquetas
            aug_label_name = f"{base_name}_aug_{i}.txt"
            with open(
                os.path.join(LABELS_AUMENTADAS_DIR, aug_label_name), "w"
            ) as f_out:
                for bbox, class_id in zip(aug_bboxes, augmented["class_labels"]):
                    x, y, w, h = bbox
                    f_out.write(f"{class_id} {x} {y} {w} {h}\n")

    print(
        f"\n¡Aumentación completada! Se han creado {len(image_files) * NUM_AUMENTACIONES_POR_IMAGEN} nuevas imágenes y etiquetas."
    )


if __name__ == "__main__":
    augment_dataset()
