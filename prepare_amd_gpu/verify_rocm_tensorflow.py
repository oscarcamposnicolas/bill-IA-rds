import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")

# Intentar configurar la memoria de la GPU para evitar problemas de asignación comunes
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Configura el crecimiento de memoria para todas las GPUs disponibles
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth set for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        # El crecimiento de memoria debe establecerse antes de que las GPUs hayan sido inicializadas
        print(f"Error setting memory growth: {e}")

print("Num GPUs Available: ", len(gpus))

if len(gpus) > 0:
    print("GPU(s) detected:", gpus)
    for i, gpu_device in enumerate(gpus):
        print(f"--- Details for GPU {i} ---")
        print(f"  Name: {gpu_device.name}")
        """
        try:
            print(f"  Attempting operation on GPU {i}...")
            with tf.device(gpu_device.name):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"  Matrix multiplication successful on GPU {i}:\n{c.numpy()}")
        except RuntimeError as e:
            print(f"  Error during GPU {i} operation test: {e}")
        """
else:
    print("No GPU detected by TensorFlow.")
    print(
        "Ensure ROCm is correctly installed, environment variables (PATH, LD_LIBRARY_PATH) point to ROCm,"
    )
    print("and your TensorFlow version supports your ROCm version.")

# Información adicional de la build de TensorFlow
print(
    f"Is TensorFlow built with CUDA support? {tf.test.is_built_with_cuda()}"
)  # Esperamos False o que no importe si ROCm es True
print(
    f"Is TensorFlow built with ROCm support? {tf.test.is_built_with_rocm()}"
)  # ¡Esperamos True!

from tensorflow.python.framework import test_util

if tf.config.list_physical_devices("GPU"):  # Usando la forma actualizada
    print(
        "TensorFlow reports GPU is available via tf.config.list_physical_devices('GPU')."
    )
else:
    print(
        "TensorFlow reports GPU is NOT available via tf.config.list_physical_devices('GPU')."
    )
