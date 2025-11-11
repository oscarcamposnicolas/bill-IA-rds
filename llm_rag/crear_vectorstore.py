"""
Módulo de Creación de la Base de Datos Vectorial (Vector Store) (Fase 7, Paso 1).

Este script inicializa el componente de Memoria Externa para el LLM, utilizando
la arquitectura RAG (Retrieval-Augmented Generation). Su función es procesar
la documentación del proyecto (reglas, especificaciones) y convertirla en un
formato vectorial indexado para consultas semánticas.

Propósito principal:
1.  Habilitar el LLM para responder preguntas específicas y complejas sobre el billar
    Bola 9, yendo más allá de su conocimiento general.
2.  Crear la base de datos de conocimiento persistente del sistema experto.
"""

import os
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
# Ruta a la carpeta que contiene los documentos PDF
PDFS_PATH = "llm_rag/corpus_billar/"
# Nombre para la base de datos vectorial que vamos a crear
VECTORSTORE_NAME = "llm_rag/billar_expert_db"
# Modelo de embeddings. 'paraphrase-multilingual-MiniLM-L12-v2' es bueno para múltiples idiomas, incluido el español.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def crear_base_de_datos_vectorial():
    """
    Función principal para procesar los PDFs y crear la base de datos vectorial FAISS.
    """
    print("Iniciando la creación de la base de datos vectorial...")
    start_time = time.time()

    # 1. Cargar todos los documentos de la carpeta
    print(f"Cargando documentos desde la carpeta: '{PDFS_PATH}'")
    documents = []
    for file in os.listdir(PDFS_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDFS_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    if not documents:
        print("¡Error! No se encontraron documentos PDF en la carpeta especificada.")
        return

    print(
        f"Cargados {len(documents)} páginas de {len(os.listdir(PDFS_PATH))} archivos PDF."
    )

    # 2. Dividir los documentos en fragmentos (chunks)
    print("Dividiendo los documentos en fragmentos de texto...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(text_chunks)} fragmentos.")

    # 3. Inicializar el modelo de embeddings
    print(f"Cargando el modelo de embeddings: '{EMBEDDING_MODEL}'...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Crear la base de datos vectorial FAISS y almacenar los embeddings
    print("Creando la base de datos vectorial con FAISS...")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    # 5. Guardar la base de datos localmente
    print(f"Guardando la base de datos en: '{VECTORSTORE_NAME}'...")
    vectorstore.save_local(VECTORSTORE_NAME)

    end_time = time.time()
    print("\n--- ¡Proceso Completado! ---")
    print(
        f"La base de datos vectorial '{VECTORSTORE_NAME}' ha sido creada y guardada con éxito."
    )
    print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos.")

    # Hacer una prueba rápida de búsqueda
    print("\nRealizando una búsqueda de prueba...")
    test_query = "Cual es la regla para una falta?"
    results = vectorstore.similarity_search(test_query, k=2)
    print(f"Resultados para la consulta: '{test_query}'")
    for doc in results:
        print(
            f"\n--- Fragmento Relevante (de '{os.path.basename(doc.metadata.get('source', 'N/A'))}')---"
        )
        print(doc.page_content)
        print("--------------------------------------------------")


if __name__ == "__main__":
    crear_base_de_datos_vectorial()
