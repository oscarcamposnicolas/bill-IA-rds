"""
Módulo de Aumentación de Dataset (Black Edition) (Fase 3, Paso 1).

Este script aplica un pipeline de transformaciones geométricas y de color
a las imágenes y anotaciones del dataset de billar.

El objetivo es incrementar artificialmente el tamaño del dataset de entrenamiento
y exponer el modelo YOLO a variaciones de iluminación, perspectiva y escala,
lo cual es vital para la robustez en el entorno real.

Se utiliza la librería Albumentations para manejar la compleja recalculación
de las bounding boxes tras cada transformación.
"""

import os
import random

import albumentations as A
import cv2

# --- CONFIGURACIÓN ---
# Directorios de origen (60 imágenes y etiquetas corregidas)
IMAGENES_ORIGINALES_DIR = "Proyecto_Bolas_LS/imagenes/"
LABELS_ORIGINALES_DIR = "Proyecto_Bolas_LS/final_yolo_labels/"

# Directorios de salida para los datos aumentados
IMAGENES_AUMENTADAS_DIR = "Proyecto_Bolas_LS/dataset_be_aumentado/images/"
LABELS_AUMENTADAS_DIR = "Proyecto_Bolas_LS/dataset_be_aumentado/labels/"

# Número de versiones nuevas a crear por cada imagen original
NUM_AUMENTACIONES_POR_IMAGEN = 15
# --- FIN DE LA CONFIGURACIÓN ---


def augment_dataset():
    # Crear directorios de salida
    os.makedirs(IMAGENES_AUMENTADAS_DIR, exist_ok=True)
    os.makedirs(LABELS_AUMENTADAS_DIR, exist_ok=True)

    # Definir el pipeline de aumentación.
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
        label_path = os.path.join(LABELS_ORIGINALES_DIR, base_name + ".txt")

        if not os.path.exists(label_path):
            continue

        # Cargar imagen y etiquetas
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id, x, y, w, h = map(float, parts)
                bboxes.append([x, y, w, h])
                class_labels.append(int(class_id))

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
