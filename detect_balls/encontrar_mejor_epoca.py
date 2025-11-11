"""
Módulo de Análisis Post-Entrenamiento y Selección de Modelo Óptimo (Fase 2, Utilidad).

Este script analiza el rendimiento de un modelo YOLO a lo largo de su entrenamiento
para identificar la 'mejor época' (best epoch). La mejor época se define como el
punto de entrenamiento donde se logra la máxima precisión (Mean Average Precision, mAP),
optimizando así el equilibrio entre aprendizaje y generalización, y previniendo
el sobreajuste (overfitting).

Este módulo requiere que el entrenamiento haya guardado el archivo 'results.csv'.
"""

import pandas as pd

# --- CONFIGURACIÓN ---
# Ruta al archivo results.csv del último entrenamiento
RUTA_RESULTS_CSV = "./detect_balls/runs/Modelo_Hibrido_v1/results.csv"
# --- FIN DE LA CONFIGURACIÓN ---


def encontrar_mejor_epoca():
    try:
        # Cargar los resultados en un DataFrame de pandas
        df = pd.read_csv(RUTA_RESULTS_CSV)

        # Limpiar los nombres de las columnas por si tienen espacios extra
        df.columns = df.columns.str.strip()

        # Nombre de la métrica que define el "mejor" modelo
        metrica_clave = "metrics/mAP50-95(B)"

        # Encontrar el índice de la fila con el valor máximo en esa métrica
        indice_mejor_epoca = df[metrica_clave].idxmax()

        # Obtener toda la información de esa fila (de esa época)
        mejor_epoca_info = df.loc[indice_mejor_epoca]

        # El número de la época es el índice + 1
        # mejor_epoca_num = int(mejor_epoca_info["epoch"]) + 1
        mejor_epoca_num = int(mejor_epoca_info["epoch"])
        mejor_map50_95 = mejor_epoca_info[metrica_clave]
        mejor_map50 = mejor_epoca_info["metrics/mAP50(B)"]

        print("\n--- Análisis del Mejor Modelo (`best.pt`) ---")
        print(f"El mejor rendimiento se alcanzó en la ÉPOCA: {mejor_epoca_num}")
        print(f"  - mAP50-95 (estricto): {mejor_map50_95:.4f}")
        print(f"  - mAP50 (estándar):   {mejor_map50:.4f}")
        print("\nEl archivo 'best.pt' corresponde a los pesos guardados en esa época.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta '{RUTA_RESULTS_CSV}'.")
        print("Asegúrate de que la ruta a tu carpeta de 'runs' es correcta.")
    except KeyError:
        print(f"Error: No se encontró la columna '{metrica_clave}' en el archivo .csv.")
        print("Asegúrate de que el nombre de la métrica es correcto.")


if __name__ == "__main__":
    encontrar_mejor_epoca()
