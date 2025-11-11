"""
Módulo de Verificación de Entorno (Fase 1: Preparación de GPU).

Este script verifica la disponibilidad de aceleración por GPU para el framework TensorFlow
en sistemas que utilizan la plataforma ROCm (Radeon Open Compute) de AMD.
Es una prueba esencial para confirmar la configuración del entorno virtual.
"""

import platform

import tensorflow as tf


def verify_tensorflow_rocm_support():
    """
    Verifica y reporta la disponibilidad de dispositivos GPU y los detalles
    de la compilación de TensorFlow.

    Returns:
        str: Un resumen del estado de soporte de la GPU y la versión del sistema.
    """

    # --- 1. Información General del Sistema ---
    print("===================================================================")
    print("           Diagnóstico de Soporte TensorFlow/ROCm                  ")
    print("===================================================================")
    print(f"Versión del Sistema Operativo: {platform.platform()}")
    print(f"Versión de Python: {platform.python_version()}")
    print(f"Versión de TensorFlow: {tf.__version__}")

    # --- 2. Verificación de Aceleración por GPU ---

    # tf.config.list_physical_devices('GPU') es el método estándar para listar hardware
    # disponible y visible para TensorFlow.
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print(
            "\n¡Aceleración por GPU detectada! Dispositivos visibles para TensorFlow:"
        )
        for gpu in gpus:
            print(f"   - Dispositivo encontrado: {gpu.name} | Tipo: {gpu.device_type}")

        # Opcional: Mostrar que TensorFlow fue compilado con soporte para XLA (ROCm/CUDA)
        print("\nInformación de compilación de TensorFlow:")
        print(
            f"   - Soporte para compilación XLA (Aceleración): {tf.config.experimental.list_devices('XLA_GPU')}"
        )
    else:
        print("\nAceleración por GPU NO detectada.")
        print(
            "   - TensorFlow está utilizando la CPU para el entrenamiento y la inferencia."
        )

    print("\n===================================================================")
    return gpus


# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    verify_tensorflow_rocm_support()
