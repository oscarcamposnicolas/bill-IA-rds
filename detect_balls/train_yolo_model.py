import os

from ultralytics import YOLO

# --- 1. Definir rutas y parámetros ---

# data_yaml_file = "custom_data.yaml"
data_yaml_file = "custom_data_shared_labels.yaml"

base_project_dir = os.path.join(".", "detect_balls")
data_yaml_path = os.path.join(base_project_dir, data_yaml_file)
execution_name = "Modelo_Hibrido_v1"
execution_epochs = 150
execution_image_resize = 640
execution_batch = 24
early_stop = 30
tasa_aprendizaje = 0.01

# Directorio donde se guardarán los resultados del entrenamiento (pesos, logs, etc.)
runs_dir = os.path.join(base_project_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)  # Crea el directorio si no existe

# --- 2. Cargar un modelo pre-entrenado (ej. YOLOv8n) ---
# 'n' es por nano, la versión más pequeña y rápida para empezar.

# print("Cargando modelo YOLOv8n pre-entrenado...")
# model = YOLO(os.path.join(base_project_dir, "yolo8", "yolov8n.pt"))

# print("Cargando modelo YOLO11n pre-entrenado...")
# model = YOLO(os.path.join(base_project_dir, "yolo11", "yolo11n.pt"))

# print("Cargando modelo YOLO11m pre-entrenado...")
model = YOLO(os.path.join(base_project_dir, "models_yolo11", "yolo11m.pt"))

# print("Cargando modelo BEST classic pool balls: pool_classic.pt...")
# model = YOLO(os.path.join(base_project_dir, "models_custom", "pool_classic.pt"))

# print("Cargando modelo best.pt del ultimo entrenamiento...")
# model = YOLO(os.path.join(runs_dir, execution_name, "weights", "last.pt"))


# print("Cargando modelo best.pt del ultimo entrenamiento...")
# model = YOLO(os.path.join(base_project_dir, "poolballs46.pt"))
# model = YOLO(os.path.join(runs_dir, execution_name, "weights", "best.pt"))

# --- 3. Entrenar el modelo ---
# Documentación de los argumentos de entrenamiento: https://docs.ultralytics.com/usage/train/

# =========================================================================
# FASE 1: ENTRENAR SOLO LA "CABEZA" (CONGELANDO EL CONOCIMIENTO BASE)
# =========================================================================
"""
print("--- INICIANDO FASE 1: ENTRENAMIENTO DE LA CABEZA (BACKBONE CONGELADO) ---")

# Cargar el modelo experto en bolas clásicas
model_fase1 = YOLO(os.path.join(base_project_dir, "models_custom", "pool_classic.pt"))

# El parámetro clave aquí es 'freeze=11'. Congela las primeras 11 capas del modelo.
# Usamos menos épocas y paciencia porque solo queremos estabilizar la nueva capa de clasificación.
results_fase1 = model_fase1.train(
    data=data_yaml_path,
    epochs=50,
    patience=15,
    batch=24,
    imgsz=640,
    name="Supermodelo_Fase1_Head",
    project=runs_dir,
    freeze=11,  # ¡Parámetro clave! Congela las capas del backbone.
)

print("\n--- FASE 1 COMPLETADA ---")

# Obtenemos la ruta al mejor modelo de la Fase 1
# La variable results.save_dir contiene la ruta a la carpeta de la ejecución
path_fase1_best = os.path.join(results_fase1.save_dir, "weights/best.pt")
print(f"Mejor modelo de la Fase 1 guardado en: {path_fase1_best}")
"""

# =========================================================================
# FASE 2: AJUSTE FINO DE TODO EL MODELO (CON LEARNING RATE BAJO)
# =========================================================================
"""
print("\n--- INICIANDO FASE 2: AJUSTE FINO COMPLETO (LEARNING RATE BAJO) ---")

# Cargar el modelo resultante de la Fase 1
#model_fase2 = YOLO(os.path.join(runs_dir, "Supermodelo_Fase1_Head2", "weights", "best.pt"))

# Ahora entrenamos todo el modelo (sin 'freeze') pero con una tasa de aprendizaje muy baja
results_fase2 = model_fase2.train(
    data=data_yaml_path,
    epochs=150,
    patience=30,
    batch=24,
    imgsz=640,
    name="Supermodelo_Fase2_Final",
    project=runs_dir,
    augment=True,  # Mantenemos las aumentaciones que ya tenías
    mixup=0.1,
    hsv_s=0.9,
    lr0=0.0005,  # ¡Parámetro clave! Tasa de aprendizaje extremadamente baja.
    optimizer="AdamW",  # ¡NUEVO! Forzamos el optimizador y desactivamos el modo 'auto'
)

print("\n--- ¡ENTRENAMIENTO POR FASES COMPLETADO! ---")
path_fase2_best = os.path.join(results_fase2.save_dir, "weights/best.pt")
print(f"El 'Supermodelo' final está listo en: {path_fase2_best}")
"""

# =========================================================================
# ENTRENAMIENTO COMPLETO DESDE VARIABLES
# =========================================================================
print("\nIniciando entrenamiento del modelo...")

results = model.train(
    data=data_yaml_path,  # Ruta a tu archivo de configuración del dataset
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
