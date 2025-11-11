# 🎱 bill-IA-rds: Sistema de IA Híbrido y On-Premise para Análisis de Billar

> Sistema de IA (100% on-premise) para billar pool. Integra Visión Clásica (Heurística Robusta + Homografía), Deep Learning (YOLO para bolas/troneras) y ML Clásico (RF para contexto). Resuelve la **orientación semántica** de la mesa y genera **coordenadas precisas para análisis de juego**. Incluye motor de reglas RAG (LLM) local **para consulta experta**.

Este repositorio contiene el código fuente completo y la documentación del proyecto `bill-IA-rds`, un sistema de Visión por Computadora y IA Híbrida diseñado para el análisis, arbitraje y consulta de reglas del juego de billar pool.

## 1\. 📖 Documentación Completa (Wiki del Proyecto)

Toda la metodología, la evolución del proyecto, el análisis técnico de cada script y la justificación de las decisiones de ingeniería (I+D) se encuentran documentados en el **Wiki oficial del repositorio**.

### [➡️ Archivo de la presentacion del Proyecto de deteccion de bolas ⬅️](https://github.com/oscarcamposnicolas/bill-IA-rds/blob/main/fases_html/bloque0.html)

### [➡️ Accede al Wiki del Proyecto aquí ⬅️](https://github.com/oscarcamposnicolas/bill-IA-rds/wiki)

El Wiki está estructurado por fases, replicando la cronología del desarrollo del proyecto:

  * **Fase 1:** Preparación del Entorno GPU
  * **Fase 2:** Detección de Bolas (P1)
  * **Fase 3:** Pipeline de Datos (Label Studio)
  * **Fase 4:** Clasificador Híbrido (RF)
  * **Fase 5:** Detección de Mesa (P4)
  * **Fase 6:** Detección de Troneras (P1.5)
  * **Fase 7:** Sistema Experto (RAG)
  * **Fase 8:** Arquitectura Final y Próximos Pasos

## 2\. 🏛️ Arquitectura del Sistema (Pipeline Híbrido)

El proyecto utiliza un *pipeline* de inferencia modular que orquesta múltiples modelos de IA y algoritmos de Visión Clásica (CV) para construir una comprensión completa de la escena.

1.  **Detección de Mesa (P4):** Una heurística de CV (Área Mínima Válida) localiza las 4 esquinas del fieltro.
2.  **Detección de Troneras (P1.5):** Un modelo YOLO especializado detecta las 6 troneras.
3.  **Análisis de Orientación:** Un módulo geométrico compara las esquinas (P4) y las troneras (P1.5) para determinar la orientación real (H/V) de la mesa.
4.  **Cálculo de Homografía:** Se genera la Matriz $H$ para mapear a un plano cenital de 1000x500.
5.  **Detección de Bolas y Contexto (P1):** Un modelo YOLO híbrido (`pool_hybrid.pt`) detecta las bolas, y un clasificador Random Forest (`context_classifier.joblib`) etiqueta la escena (ej. "Classic").
6.  **Salida Final:** Las coordenadas de las bolas (P1) se multiplican por la Matriz $H$ para obtener las coordenadas finales en el plano de la mesa.

## 3\. 📦 Contenido del Repositorio (Nota Importante)

Para mantener el repositorio ágil y enfocado en el código fuente, **este repositorio NO incluye los siguientes artefactos pesados**:

  * **Datasets de Imágenes:** Los conjuntos de datos de entrenamiento, validación y prueba (que ocupan varios GB) no están incluidos.
  * **Modelos Entrenados:** Los archivos de pesos (`.pt`, `.joblib`) no están incluidos.

Todo el **código fuente para generar estos artefactos** (scripts de aumentación, conversión de Label Studio y entrenamiento de modelos) está incluido en las carpetas `detect_balls/`, `detect_pockets/`, etc., permitiendo la **reproducibilidad completa** del proyecto.

## 4\. 🚀 Inicio Rápido y Demo del Proyecto (Flask)

Este repositorio incluye la **aplicación web Flask funcional** que se presentó al final del curso. Esta demo sirve como prueba de concepto interactiva y como documentación navegable.

**Componentes Clave de la Demo:**

  * `app.py`: El servidor Flask que gestiona las rutas y la lógica de la API.
  * `motor_inferencia.py`: El *pipeline* híbrido (YOLO + RF) para la detección de bolas y contexto.
  * `templates/inferencia.html`: La página principal de la aplicación, que permite **probar la inferencia** subiendo una imagen.
  * `fases_html/` (y `fases_html/bloque0.html`): Archivos HTML estáticos que sirven como **documentación interactiva** de la evolución del proyecto.

### Ejecución de la Demo y la presentacion de **deteccion de bolas**:

1.  **Clonar:**

    ```bash
    git clone https://github.com/tu-usuario/bill-IA-rds.git
    cd bill-IA-rds
    ```

2.  **Entorno Virtual:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instalar Dependencias:**

    ```bash
    pip install -r requirements.txt
    ```
    
    *(Nota: La configuración de aceleración por GPU (CUDA/ROCm) para PyTorch debe realizarse según la documentación oficial de PyTorch para tu hardware específico.)*

4.  **Ejecutar el Servidor Flask:**

    ```bash
    flask run
    ```

5.  **Probar la Inferencia:**

      * Para poder probar la inferencia, hay que generar los modelos entrenados, tanto de Deep Learning (YOLO) como de Machine Learning (Random Forest).
      * Abre `http://127.0.0.1:5000` en tu navegador para acceder a la herramienta de subida y prueba de inferencia.

6.  **Explorar la Documentación de Fases:**

      * Para navegar por la presentación de la evolución del proyecto, abre el archivo `fases_html/bloque0.html` directamente en tu navegador (ej. `file:///ruta/a/tu/proyecto/bill-IA-rds/fases_html/bloque0.html`).

## 5\. 🗺️ Fases Futuras (Fase 8: Unificación)

El estado actual del proyecto es una colección de *scripts* de I+D funcionales y una demo en Flask (Fases 1-4). Los próximos pasos se centran en la **refactorización y unificación** para la generación de una aplicación web.

1.  **Refactorización Modular:** Migrar la lógica de los *scripts* sueltos a una estructura de "expertos" en la carpeta `ia_modules/`.
2.  **Aplicación Streamlit:** Construir una nueva aplicación web (`presentation_app.py`) usando **Streamlit** que importe estos módulos y permita probar el *pipeline* unificado completo (P4 $\rightarrow$ P1.5 $\rightarrow$ P1 $\rightarrow$ Homografía).

