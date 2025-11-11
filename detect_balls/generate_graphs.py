"""
Módulo de Generación de Gráficos de Análisis (Fase 2, Utilidad).

Este script de post-procesamiento lee el archivo 'results.csv' generado por el
entrenamiento del modelo YOLO y genera un conjunto de gráficos de línea de alta
calidad (usando Matplotlib y Seaborn) para analizar la convergencia, el rendimiento
y el sobreajuste del modelo.

Se generan cuatro gráficos clave:
1.  Pérdidas de Entrenamiento (Train Losses).
2.  Métricas de Rendimiento (mAP, Precision, Recall).
3.  Pérdidas de Validación (Validation Losses).
4.  Tasas de Aprendizaje (Learning Rates).
"""

#!pip install pandas matplotlib seaborn

import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the CSV file into a pandas DataFrame
base_project_dir = os.path.join(".", "detect_balls")
csv_file = os.path.join(base_project_dir, "runs", "Modelo_Hibrido_v1", "results.csv")
df = pd.read_csv(csv_file)

# Plotting training losses
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="epoch", y="train/box_loss", label="Train Box Loss")
sns.lineplot(data=df, x="epoch", y="train/cls_loss", label="Train Class Loss")
sns.lineplot(data=df, x="epoch", y="train/dfl_loss", label="Train DFL Loss")
plt.title("Training Losses vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
# plt.show()
# Guarda la figura en un archivo PNG en la misma carpeta del script
plt.savefig(os.path.join(base_project_dir, "graph_results", "graph_train.png"))
# Cerrar la figura para liberar memoria
plt.close()

# Plotting metrics
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="epoch", y="metrics/precision(B)", label="Precision (B)")
sns.lineplot(data=df, x="epoch", y="metrics/recall(B)", label="Recall (B)")
sns.lineplot(data=df, x="epoch", y="metrics/mAP50(B)", label="mAP50 (B)")
sns.lineplot(data=df, x="epoch", y="metrics/mAP50-95(B)", label="mAP50-95 (B)")
plt.title("Metrics vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
# plt.show()
# Guarda la figura en un archivo PNG en la misma carpeta del script
plt.savefig(os.path.join(base_project_dir, "graph_results", "graph_metrics.png"))
# Cerrar la figura para liberar memoria
plt.close()

# Plotting validation losses
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="epoch", y="val/box_loss", label="Validation Box Loss")
sns.lineplot(data=df, x="epoch", y="val/cls_loss", label="Validation Class Loss")
sns.lineplot(data=df, x="epoch", y="val/dfl_loss", label="Validation DFL Loss")
plt.title("Validation Losses vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
# plt.show()
# Guarda la figura en un archivo PNG en la misma carpeta del script
plt.savefig(os.path.join(base_project_dir, "graph_results", "graph_validation.png"))
# Cerrar la figura para liberar memoria
plt.close()

# Plotting learning rates
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="epoch", y="lr/pg0", label="LR pg0")
sns.lineplot(data=df, x="epoch", y="lr/pg1", label="LR pg1")
sns.lineplot(data=df, x="epoch", y="lr/pg2", label="LR pg2")
plt.title("Learning Rate vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
# plt.show()
# Guarda la figura en un archivo PNG en la misma carpeta del script
plt.savefig(os.path.join(base_project_dir, "graph_results", "graph_learning_rates.png"))
# Cerrar la figura para liberar memoria
plt.close()
