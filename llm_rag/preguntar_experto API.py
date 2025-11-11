import os
import time
from langchain_community.vectorstores import FAISS

# Cambio recomendado por el warning: usamos la nueva librería para los embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN ---
VECTORSTORE_NAME = "llm_rag/billar_expert_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    """
    Función principal para inicializar el chatbot y responder preguntas.
    """
    # Validar que la clave de API está configurada
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
        print("Por favor, configúrala con tu clave de API de Google AI Studio.")
        return

    print("Iniciando el Asistente de Billar...")

    # 1. Cargar el modelo de embeddings (usando la nueva librería)
    print("Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Cargar la base de datos vectorial existente
    print(f"Cargando la base de datos vectorial: '{VECTORSTORE_NAME}'...")
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_NAME, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error al cargar la base de datos vectorial: {e}")
        print(
            "Asegúrate de que la carpeta 'billar_expert_db' existe y se creó correctamente."
        )
        return

    # 3. Configurar el LLM (Google Gemini)
    print("Conectando con el modelo de lenguaje (Google Gemini)...")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # 4. Crear una plantilla de prompt para guiar al LLM
    prompt_template = """
    Eres un experto mundial en billar, preciso y riguroso. Tu única fuente de conocimiento son los documentos proporcionados.
    Responde a la pregunta del usuario basándote únicamente en el siguiente contexto.
    Si la respuesta no se encuentra en el contexto, di "Lo siento, no tengo información sobre eso en mis documentos".
    No inventes nada. Sé claro y conciso.

    Contexto:
    {context}

    Pregunta del usuario:
    {question}

    Respuesta de experto:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 5. Crear la cadena de RetrievalQA
    print("Creando la cadena de respuesta...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    print("\n--- ¡Asistente listo! ---")
    print("Escribe tu pregunta o 'salir' para terminar.")

    # 6. Bucle de preguntas y respuestas
    while True:
        query = input("\nTu pregunta: ")
        if query.lower() == "salir":
            print("¡Hasta la próxima!")
            break
        if query:
            start_time = time.time()

            # Procesar la consulta con la cadena
            result = qa_chain.invoke({"query": query})

            end_time = time.time()

            # Mostrar la respuesta y las fuentes
            print("\nRespuesta del Experto:")
            print(result["result"])
            print(f"\n(Respuesta generada en {end_time - start_time:.2f} segundos)")

            # Opcional: Mostrar de qué documentos sacó la información
            print("\nFuentes consultadas:")
            # Crear un set para evitar fuentes duplicadas
            source_files = set()
            for doc in result["source_documents"]:
                source_files.add(os.path.basename(doc.metadata.get("source", "N/A")))

            for source_file in source_files:
                print(f"- {source_file}")


if __name__ == "__main__":
    main()
