"""
Módulo de Mantenimiento y Actualización de la Base de Datos Vectorial (Fase 7, Paso 1).

Este script gestiona la actualización incremental del Vector Store existente.
Su función es procesar nuevos documentos o cambios en la documentación del proyecto,
generar sus embeddings y añadirlos a la base de datos persistente (ChromaDB)
sin eliminar el conocimiento previo.

Propósito principal:
1.  Eficiencia: Evitar la costosa recreación completa de la Vector Store.
2.  Mantenimiento: Asegurar que el LLM experto siempre tenga acceso al conocimiento
    más reciente del proyecto (ej., nuevas reglas de juego o especificaciones técnicas).
"""

import os
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURACIÓN ---
PDFS_PATH = "llm_rag/corpus_billar/"
VECTORSTORE_NAME = "llm_rag/billar_expert_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# Archivo para registrar los documentos ya procesados
PROCESSED_LOG_FILE = "llm_rag/processed_files.log"


def cargar_archivos_procesados():
    """Carga la lista de archivos que ya están en la base de datos."""
    if not os.path.exists(PROCESSED_LOG_FILE):
        return set()
    with open(PROCESSED_LOG_FILE, "r") as f:
        return set(line.strip() for line in f)


def guardar_archivos_procesados(nuevos_archivos):
    """Añade los nuevos archivos procesados al registro."""
    with open(PROCESSED_LOG_FILE, "a") as f:
        for archivo in nuevos_archivos:
            f.write(archivo + "\n")


def actualizar_base_de_datos():
    """
    Función que detecta nuevos PDFs y los añade a la base de datos FAISS existente.
    """
    print("Iniciando el script de actualización de la base de datos...")

    # 1. Identificar archivos nuevos
    archivos_procesados = cargar_archivos_procesados()
    archivos_actuales = {f for f in os.listdir(PDFS_PATH) if f.endswith(".pdf")}
    archivos_nuevos = list(archivos_actuales - archivos_procesados)

    if not archivos_nuevos:
        print(
            "No se encontraron nuevos documentos para añadir. La base de datos ya está actualizada."
        )
        return

    print(
        f"Se han encontrado {len(archivos_nuevos)} nuevos documentos para procesar: {archivos_nuevos}"
    )

    # 2. Cargar los nuevos documentos
    print("Cargando contenido de los nuevos documentos...")
    nuevos_documentos = []
    for file in archivos_nuevos:
        pdf_path = os.path.join(PDFS_PATH, file)
        loader = PyPDFLoader(pdf_path)
        nuevos_documentos.extend(loader.load())

    # 3. Procesar los documentos (dividir en chunks)
    print("Dividiendo los nuevos documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(nuevos_documentos)
    print(f"Se han creado {len(text_chunks)} nuevos fragmentos.")

    # 4. Cargar el modelo de embeddings y la base de datos existente
    print("Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Cargando la base de datos vectorial existente: '{VECTORSTORE_NAME}'...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_NAME, embeddings, allow_dangerous_deserialization=True
    )

    # 5. Añadir los nuevos fragmentos a la base de datos
    print("Añadiendo nuevos fragmentos a la base de datos vectorial...")
    start_time = time.time()
    vectorstore.add_documents(text_chunks)
    end_time = time.time()
    print(f"Fragmentos añadidos en {end_time - start_time:.2f} segundos.")

    # 6. Guardar la base de datos actualizada y el registro
    vectorstore.save_local(VECTORSTORE_NAME)
    guardar_archivos_procesados(archivos_nuevos)

    print("\n--- ¡Actualización Completada! ---")
    print(
        f"La base de datos '{VECTORSTORE_NAME}' ha sido actualizada con los nuevos documentos."
    )


if __name__ == "__main__":
    # La primera vez, necesitamos crear el log con los archivos iniciales.
    # Descomentar y ejecuta esta parte SOLO si nunca has creado el 'processed_files.log'
    # if not os.path.exists(PROCESSED_LOG_FILE):
    #   print("Creando archivo de registro inicial...")
    #   initial_files = [f for f in os.listdir(PDFS_PATH) if f.endswith(".pdf")]
    #   guardar_archivos_procesados(initial_files)
    #   print("Registro inicial creado con los archivos existentes.")

    actualizar_base_de_datos()
