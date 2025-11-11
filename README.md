# 🎱 bill-IA-rds: Sistema Híbrido de Análisis y Arbitraje para Billar Pool

Sistema de IA (100% on-premise) para billar pool. 

Integra Visión Clásica (Heurística Robusta + Homografía), Deep Learning (YOLO para bolas/troneras) y ML Clásico (RF para contexto). 

Resuelve la orientación semántica de la mesa y genera coordenadas precisas para análisis de juego. 

Incluye motor de reglas RAG (LLM) local para consulta experta.

## 1. Resumen del Proyecto

`bill-IA-rds` es un sistema de Inteligencia Artificial diseñado para realizar un análisis completo de una mesa de billar pool a partir de una sola imagen. A diferencia de un simple detector de objetos, este proyecto resuelve problemas complejos de **visión por computadora, perspectiva y clasificación de contexto** mediante una arquitectura híbrida.

El sistema final es capaz de:

- **Detectar la Mesa (P4):** Localizar la superficie de juego y sus 4 esquinas, incluso en imágenes con ruido visual y bordes exteriores confusos.
- **Resolver la Orientación (P1.5):** Determinar la orientación real ("larga" vs. "corta") de la mesa, solucionando la ambigüedad de la perspectiva.
- **Geolocalizar Objetos (P1):** Detectar todas las bolas, identificar sus clases y mapearlas desde coordenadas de píxeles 2D a un plano cenital 2D ideal.
- **Clasificar el Contexto (Fase 4):** Determinar qué conjunto de reglas se aplica (ej. 'Clásico' vs 'Black Edition') basándose en las bolas detectadas.
- **Consultar Reglas (Fase 7):** Responder preguntas de lenguaje natural sobre las reglas del juego usando un *pipeline* RAG.

## 2. Arquitectura del Sistema (Pipeline de Inferencia)

El núcleo del proyecto es un *pipeline* de inferencia modular que orquesta múltiples modelos de IA y algoritmos de Visión Clásica (CV) en un orden específico para construir una comprensión completa de la escena.

**Flujo de Datos de Inferencia:**

1. **Entrada:** Una única imagen (`.jpg`/`.png`) de una mesa de billar.
2. **Módulo P4 (Detector de Mesa):**
    - La imagen es procesada por Canny y Hough (`find_table_borders.py`).
    - La **Heurística de "Área Mínima Válida"** (`filter_table_borders.py`) filtra cientos de líneas candidatas para encontrar las 4 esquinas del fieltro.
3. **Módulo P1.5 (Detector de Troneras):**
    - Un modelo YOLO especializado (`pocket_detector.pt`) detecta las 6 troneras (`pocket_corner`, `pocket_side`).
4. **Módulo Geométrico (Orientación):**
    - Compara la ubicación de las **esquinas (P4)** con la ubicación de las **troneras (P1.5)** para determinar semánticamente la orientación real (Horizontal/Vertical) de la mesa, resolviendo el fallo de la perspectiva.
5. **Módulo Geométrico (Homografía):**
    - Calcula la Matriz de Homografía ($H$) usando las 4 esquinas y la orientación detectada, creando un mapeo a un lienzo estándar de 1000x500 (`perspective_transform.py`).
6. **Módulo P1 (Detector de Bolas y Contexto):**
    - Un modelo YOLO híbrido (`pool_hybrid.pt`) detecta todas las bolas en la imagen (`detect_balls`).
    - Un clasificador Random Forest (`context_classifier.joblib`) analiza el *conteo* de bolas detectadas para clasificar la escena (ej. "Classic").
7. **Salida Final:**
    - Los centroides de las bolas (P1) se multiplican por la Matriz $H$ (P4/P1.5) para obtener las **coordenadas finales (x, y)** en el plano cenital (`balls_coord_transform.py`).

## 3. Metodología y Evolución del Proyecto

Esta sección detalla la cronología de Investigación y Desarrollo (I+D), demostrando la evolución de las soluciones técnicas implementadas.

### Fase 1: Preparación del Entorno

- **Descripción:** Configuración inicial del entorno de desarrollo para soportar aceleración por GPU (AMD ROCm) tanto para PyTorch (YOLO) como para TensorFlow.
- **Scripts Clave:** `verify_rocm_pytorch.py`, `verify_rocm_tensorflow.py`.

### Fase 2: Detección de Bolas (P1)

- **Descripción:** Desarrollo del *pipeline* de Deep Learning para la detección de bolas. Incluye el entrenamiento del modelo YOLOv11 (`pool_hybrid.pt`) y utilidades de análisis post-entrenamiento para validar la precisión y seleccionar el mejor *checkpoint*.
- **Scripts Clave:** `train_yolo_model.py`, `test_model_folder.py`, `encontrar_mejor_epoca.py`, `generate_graphs.py`.

### Fase 3: Pipeline de Datos (Label Studio)

- **Descripción:** Creación de un *pipeline* de anotación de datos robusto y automatizado. Incluye la generación de tareas para Label Studio, la conversión del formato JSON de Label Studio a formato YOLO, y la aumentación de datos (Albumentations) para generar un dataset de entrenamiento de alta calidad.
- **Scripts Clave:** `generar_json_para_label_studio.py`, `convertir_ls_a_yolo.py`, `aumentar_dataset_be.py`, `verificar_etiquetas_yolo.py`.

### Fase 4: Clasificador Híbrido (YOLO + RF)

- **Descripción:** Una innovación clave del proyecto. Para clasificar el "contexto" de la mesa (ej. 'Black Edition' vs 'Classic'), se implementó un sistema híbrido.
    1. **Ingeniería de Features:** Se utiliza la salida de YOLO (conteo de bolas por clase) para crear un **Meta-Dataset** (`generar_meta_dataset.py`).
    2. **Entrenamiento de ML Clásico:** Se entrena un clasificador **Random Forest** (Scikit-learn) sobre este Meta-Dataset (`entrenar_clasificador_contexto.py`).
- **Scripts Clave:** `inferencia_sistema_completo.py` (valida el *pipeline* híbrido).

### Fase 5: Detección de Mesa (P4)

- **Descripción:** Esta fase fue una de las más complejas. El objetivo era encontrar las 4 esquinas del fieltro.
    - **Intento 1 (Manual):** Se validó la matemática de Homografía usando selección manual (`select_table_corners.py`).
    - **Intento 2 (Filtro Simple):** Se intentó filtrar las líneas de Hough por longitud o área máxima (`Fracaso`). Esto seleccionaba el borde exterior de la madera.
    - **Solución (Heurística de "Área Mínima Válida"):** Se desarrolló una heurística robusta (`filter_table_borders.py`) que (1) filtra el ruido pequeño (<10% del área) y (2) selecciona el cuadrilátero **válido más pequeño** (el fieltro), que siempre es más pequeño que el borde de la madera.
- **Scripts Clave:** `find_table_borders.py` (Sensor), `filter_table_borders.py` (Cerebro).

### Fase 6: Detección de Troneras (P1.5)

- **Descripción:** Durante la Fase 5, se descubrió que la perspectiva hacía imposible determinar la orientación de la mesa (largo vs. corto) solo con las líneas.
    - **Solución:** Se entrenó un **segundo modelo YOLO especializado** (`pocket_detector.pt`) con el único fin de detectar `pocket_corner` y `pocket_side`.
    - **Metodología:** Se replicó el *pipeline* de datos de la Fase 3 (`convertir_ls_a_yolo.py`, `aumentar_dataset_pockets.py`) para crear y validar este nuevo modelo.
- **Scripts Clave:** `train_yolo_model.py` (Pockets), `test_model_folder.py` (Pockets).

### Fase 7: Sistema Experto (RAG/LLM)

- **Descripción:** Se añadió un componente de IA Generativa para responder preguntas sobre las reglas del billar. Se creó una base de datos vectorial (ChromaDB) a partir de la documentación del proyecto (`crear_vectorstore.py`).
- **Scripts Clave:** `preguntar_experto.py` (consulta), `preguntar_experto_API.py` (despliegue en API).

## 4. Próximos Pasos (Fase 8: Refactorización y Unificación)

El trabajo actual se centra en la Fase 8, que prepara el proyecto para la presentación final.

1. **Refactorización Modular:**
    - Migrar la lógica de los *scripts* sueltos de las Fases 2-7 a una estructura de módulos "expertos" en la carpeta `ia_modules/` (ej. `table_detector.py`, `ball_detector.py`, `geometry_transformer.py`).
2. **Implementación de la Web de Presentación:**
    - Construir la aplicación web (`presentation_app.py`) usando **Streamlit**.
    - La aplicación importará los módulos de `ia_modules/` para ejecutar el *pipeline* completo.
    - La interfaz permitirá al jurado probar las funcionalidades individuales (como se solicita en la lista de Fases) y ejecutar el *pipeline* unificado.

## 5. Instalación y Dependencias

El proyecto requiere **Python 3.10+** y las librerías listadas en `requirements.txt`.

1. **Clonar:** `git clone https://github.com/oscarcamposnicolas/bill-IA-rds.git`
2. **Entorno:** `python3 -m venv .venv && source .venv/bin/activate`
3. **Instalar:** `pip install -r requirements.txt`
4. **Ejecutar Pruebas:**
    - Validar P4: `python detect_table/pygame_table_borders.py`
    - Validar P1.5: `python detect_pockets/test_model_folder.py`
    - Validar P1: `python detect_balls/RandomForestClassifier/inferencia_sistema_completo.py`
