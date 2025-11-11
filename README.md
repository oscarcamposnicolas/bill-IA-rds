# 🎱 bill-IA-rds: Sistema de IA Híbrido y On-Premise para Análisis de Billar

> Sistema de IA (100% on-premise) para billar pool. Integra Visión Clásica (Heurística Robusta + Homografía), Deep Learning (YOLO para bolas/troneras) y ML Clásico (RF para contexto). Resuelve la **orientación semántica** de la mesa y genera **coordenadas precisas para análisis de juego**. Incluye motor de reglas RAG (LLM) local **para consulta experta**.

Este repositorio contiene el código fuente completo y la documentación del proyecto `bill-IA-rds`, un sistema de Visión por Computadora y IA Híbrida diseñado para el análisis, arbitraje y consulta de reglas del juego de billar pool.

## 1\. 📖 Documentación Completa (Wiki del Proyecto)

Toda la metodología, la evolución del proyecto, el análisis técnico de cada script y la justificación de las decisiones de ingeniería (I+D) se encuentran documentados en el **Wiki oficial del repositorio**.

### [➡️ Archivo de la presentacion del Proyecto de deteccion de bolas ⬅️](https://github.com/oscarcamposnicolas/bill-IA-rds/blob/main/fases_html/bloque0.html)

### [➡️ Accede al Wiki del Proyecto aquí ⬅️](https://www.google.com/search?q=https://github.com/tu-usuario/bill-IA-rds/wiki)

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

## 3\. 🚀 Inicio Rápido

Este proyecto está diseñado para ser 100% on-premise. Se requiere **Python 3.10+** y un entorno virtual.

### 1\. Clonar el Repositorio

```bash
git clone https://github.com/oscarcamposnicolas/bill-IA-rds.git
cd bill-IA-rds
```

### 2\. Crear y Activar el Entorno Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3\. Instalar Dependencias

```bash
# Instalar todas las librerías de Python necesarias
pip install -r requirements.txt
```

*(Nota: La configuración de aceleración por GPU (CUDA/ROCm) para PyTorch debe realizarse según la documentación oficial de PyTorch para tu hardware específico.)*

