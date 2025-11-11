"""
Módulo de Verificación de Entorno (Fase 1: PyTorch/ROCm).

Este script de diagnóstico verifica si la librería PyTorch puede acceder a la
aceleración por GPU a través de la plataforma ROCm (Radeon Open Compute) de AMD.
Es una prueba fundamental para el uso de modelos de Deep Learning que dependen
de PyTorch, como la librería Ultralytics (YOLO) o PyGame para renderizado.
"""

import platform

import torch


def verify_pytorch_rocm_support():
    """
    Ejecuta el diagnóstico de la instalación de PyTorch y su soporte para GPU.
    """

    # --- 1. Información General del Sistema y Versiones ---
    print("===================================================================")
    print("            Diagnóstico de Soporte PyTorch/ROCm                    ")
    print("===================================================================")
    print(f"Versión del Sistema Operativo: {platform.platform()}")
    print(f"Versión de Python: {platform.python_version()}")
    print(f"Versión de PyTorch: {torch.__version__}")

    # --- 2. Verificación de Aceleración por GPU (CUDA/ROCm) ---
    cuda_available = torch.cuda.is_available()

    # En PyTorch, el soporte para ROCm se maneja de forma similar a CUDA (a través de torch.cuda).
    # La variable `torch.version.hip` indica si PyTorch fue compilado con soporte para HIP/ROCm.
    rocm_hip_version = getattr(torch.version, "hip", None)
    rocm_available = cuda_available and (rocm_hip_version is not None)

    if rocm_available:
        print("\n¡Aceleración por GPU (ROCm) detectada!")
        print(f"   - Se encontró el backend HIP (ROCm). Versión: {rocm_hip_version}")
        print(f"   - Dispositivo visible: {torch.cuda.get_device_name(0)}")
        print(f"   - Capacidad de computación: {torch.cuda.get_device_capability(0)}")
    elif cuda_available:
        # Esto ocurre si es una instalación NVIDIA (CUDA)
        print("\nAceleración por GPU (CUDA) detectada.")
        print(
            "   - Esto es un indicador de que el sistema está usando NVIDIA/CUDA, no ROCm."
        )
        print(f"   - Dispositivo visible: {torch.cuda.get_device_name(0)}")
    else:
        print("\nAceleración por GPU NO detectada.")
        print(
            "   - PyTorch está utilizando la CPU para el entrenamiento y la inferencia."
        )
        print(
            "   - Verifica la instalación de los drivers AMD (ROCm) y la versión de PyTorch."
        )

    print("\n===================================================================")


# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    verify_pytorch_rocm_support()
