# unificar_y_dividir_completo.py
import os
import shutil

from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
# BE_IMAGENES_DIR = "Proyecto_Bolas_LS/imagenes/"
# BE_LABELS_DIR = "Proyecto_Bolas_LS/final_yolo_labels/"
BE_IMAGENES_DIR = "Proyecto_Bolas_LS/dataset_be_aumentado/images/"
BE_LABELS_DIR = "Proyecto_Bolas_LS/dataset_be_aumentado/labels/"

DEST_IMG_TRAIN, DEST_LBL_TRAIN = (
    "detect_balls/dataset_unificado_aumentado/images/train/",
    "detect_balls/dataset_unificado_aumentado/labels/train/",
)

os.makedirs(DEST_IMG_TRAIN, exist_ok=True)
os.makedirs(DEST_LBL_TRAIN, exist_ok=True)

DEST_IMG_VALID, DEST_LBL_VALID = (
    "detect_balls/dataset_unificado_aumentado/images/valid/",
    "detect_balls/dataset_unificado_aumentado/labels/valid/",
)

os.makedirs(DEST_IMG_VALID, exist_ok=True)
os.makedirs(DEST_LBL_VALID, exist_ok=True)

DEST_IMG_TEST, DEST_LBL_TEST = (
    "detect_balls/dataset_unificado_aumentado/images/test/",
    "detect_balls/dataset_unificado_aumentado/labels/test/",
)

os.makedirs(DEST_IMG_TEST, exist_ok=True)
os.makedirs(DEST_LBL_TEST, exist_ok=True)


# Proporciones: 10% para test, del resto, 20% para validación (total: 72% train, 18% valid, 10% test)
VALID_SIZE = 0.20
TEST_SIZE = 0.10
# --- FIN DE LA CONFIGURACIÓN ---


def main():
    print(
        "Iniciando la división (train/valid/test) y copia del dataset 'black edition' aumentado..."
    )
    image_files = [
        f
        for f in os.listdir(BE_IMAGENES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Primera división: separamos el conjunto de test
    train_val_files, test_files = train_test_split(
        image_files, test_size=TEST_SIZE, random_state=42
    )
    # Segunda división: del resto, separamos train y valid
    train_files, val_files = train_test_split(
        train_val_files, test_size=VALID_SIZE, random_state=42
    )

    print(f"Total imágenes 'black edition': {len(image_files)}")
    print(f"Añadidas a entrenamiento: {len(train_files)}")
    print(f"Añadidas a validación: {len(val_files)}")
    print(f"Añadidas a test: {len(test_files)}")

    def copy_files(file_list, dest_img_dir, dest_lbl_dir):
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + ".txt"
            shutil.copy(
                os.path.join(BE_IMAGENES_DIR, img_file),
                os.path.join(dest_img_dir, img_file),
            )
            shutil.copy(
                os.path.join(BE_LABELS_DIR, label_file),
                os.path.join(dest_lbl_dir, label_file),
            )

    copy_files(train_files, DEST_IMG_TRAIN, DEST_LBL_TRAIN)
    copy_files(val_files, DEST_IMG_VALID, DEST_LBL_VALID)
    copy_files(test_files, DEST_IMG_TEST, DEST_LBL_TEST)

    print(
        "\n¡Copia completada! El dataset unificado v2 está listo con las 3 divisiones."
    )


if __name__ == "__main__":
    main()
