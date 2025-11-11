"""
Módulo de Entrenamiento del Modelo YOLO Especializado (Fase 6, Paso 5).

Este script inicializa y ejecuta el proceso de entrenamiento del modelo de
Detección de Troneras (P1.5), utilizando la arquitectura YOLOv8n (u otra versión)
pre-entrenada y el dataset de troneras (pocket_corner, pocket_side) generado
y dividido en la Fase 6.

El objetivo principal es crear un modelo especializado de alto rendimiento para
proporcionar características semánticas (la ubicación de las 6 troneras) que son
invariantes a la perspectiva, resolviendo así el problema de la orientación de la mesa.
"""

import os

from ultralytics import YOLO

# --- 1. Definir rutas y parámetros ---

data_yaml_file = "custom_data.yaml"

base_project_dir = os.path.join(".", "detect_pockets")
data_yaml_path = os.path.join(base_project_dir, data_yaml_file)
execution_name = "detect_pockets_v1"
execution_epochs = 50
execution_image_resize = 640
execution_batch = 24
early_stop = 15
tasa_aprendizaje = 0.01

# Directorio donde se guardarán los resultados del entrenamiento (pesos, logs, etc.)
runs_dir = os.path.join(base_project_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)  # Crea el directorio si no existe

# --- 2. Cargar un modelo pre-entrenado (ej. YOLOv8n) ---
print("Cargando modelo YOLO11n pre-entrenado...")
model = YOLO(os.path.join(base_project_dir, "models_yolo11", "yolo11n.pt"))

# --- 3. Entrenar el modelo ---
# Documentación de los argumentos de entrenamiento: https://docs.ultralytics.com/usage/train/

# =========================================================================
# ENTRENAMIENTO COMPLETO DESDE VARIABLES
# =========================================================================
print("\nIniciando entrenamiento del modelo...")

results = model.train(
    data=data_yaml_path,  # Ruta al archivo de configuración del dataset
    epochs=execution_epochs,  # Número de épocas de entrenamiento (ajustable)
    imgsz=execution_image_resize,  # Tamaño de la imagen de entrada (640x640 es un buen inicio)
    batch=execution_batch,  # Tamaño del batch (ajustable, depende de tu RAM/VRAM)
    name=execution_name,  # Nombre para esta ejecución de entrenamiento
    project=runs_dir,  # Directorio raíz para guardar los resultados
    # workers=os.cpu_count() // 2   # Descomentar para usar la mitad de los núcleos de CPU para carga de datos
    # --- Control del Entrenamiento ---
    patience=early_stop,  # Parada temprana si no mejora en 25 épocas
    # --- Parámetros de Aumentación ---
    augment=True,  # Nos aseguramos de que la aumentación general esté activa
    # degrees=10.0,  # Rotación de hasta 10 grados
    # translate=0.2,  # Traslación de hasta un 20%
    # scale=0.8,  # Escalado de hasta un 80%
    # shear=5.0,  # Inclinación de hasta 5 grados
    mixup=0.1,  # Activar MixUp con un 10% de probabilidad
    hsv_s=0.9,  # Aumentar la variación de saturación
    # fliplr=0.5,  # Mantener el volteo horizontal
    # lr0=tasa_aprendizaje,
    # freeze=11,  # ¡Parámetro clave! Congela las capas del backbone.
)

print("\n¡Entrenamiento completado!")
print(f"Los resultados se guardaron en: {os.path.join(runs_dir, execution_name)}")
print("Puedes revisar los gráficos de entrenamiento y las métricas allí.")
