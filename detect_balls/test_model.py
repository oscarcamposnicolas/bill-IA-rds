"""
Módulo de Simulación de Detección (Fase 2, Utilidad).

Este script actúa como un PLACEHOLDER o un módulo de MOCK para el sistema real
de Detección de Bolas (YOLO). Su función es simular la inferencia de un modelo
de Machine Learning generando resultados aleatorios.

Propósito principal:
1. Pruebas de integración de la interfaz de usuario y del post-procesamiento.
2. Servir como módulo de fallback o prueba de concepto sin hardware de aceleración.
"""

import os
import random


def detect_billiard_balls(image_path):
    """
    Función de simulación de un modelo de detección de objetos.
    En una aplicación real, aquí es donde se cargaría el modelo (por ejemplo, con TensorFlow, PyTorch, OpenCV)
    y se procesaría la imagen para encontrar objetos.

    Args:
        image_path (str): La ruta al archivo de imagen subido.

    Returns:
        str: Un resultado de detección simulado en formato de texto.
    """
    # Comprobar si el archivo de imagen existe realmente
    if not os.path.exists(image_path):
        return "Error: El archivo de imagen no se encontró en el servidor."

    # Simulación: Generar un número aleatorio de bolas detectadas
    try:
        num_balls_detected = random.randint(3, 15)

        # Simular las coordenadas de las bolas
        results = []
        for i in range(num_balls_detected):
            ball_number = random.randint(1, 15)
            x_coord = random.randint(50, 800)
            y_coord = random.randint(50, 600)
            confidence = random.uniform(0.85, 0.99)
            results.append(
                f"- Bola #{ball_number}: encontrada en ({x_coord}, {y_coord}) con una confianza del {confidence:.2%}"
            )

        detection_summary = (
            f"Detección completada en '{os.path.basename(image_path)}'.\n"
        )
        detection_summary += (
            f"Se han encontrado un total de {num_balls_detected} bolas de billar.\n\n"
        )
        detection_summary += "Detalles:\n" + "\n".join(results)

        return detection_summary

    except Exception as e:
        return f"Ha ocurrido un error durante la simulación de la detección: {e}"
