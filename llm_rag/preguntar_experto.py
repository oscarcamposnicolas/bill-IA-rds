import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN OPTIMIZADA ---
VECTORSTORE_NAME = "llm_rag/billar_expert_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL = "gemma3:12b"


def main():
    """
    Función principal para inicializar el chatbot y responder preguntas en modo 100% local y acelerado por GPU.
    """
    print(
        f"Iniciando el Asistente de Billar en MODO LOCAL OPTIMIZADO... 🚀 (Modelo: {LOCAL_MODEL})"
    )

    # ... (El resto del código es exactamente el mismo que en la versión anterior)

    print("Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Cargando la base de datos vectorial: '{VECTORSTORE_NAME}'...")
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_NAME, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error al cargar la base de datos vectorial: {e}")
        return

    print(
        f"Conectando con el modelo local '{LOCAL_MODEL}' vía Ollama (debería usar la GPU)..."
    )
    llm = OllamaLLM(model=LOCAL_MODEL)

    prompt_template = """
    <start_of_turn>user
    Eres un experto mundial en billar, preciso y riguroso. Tu única fuente de conocimiento son los documentos proporcionados en el contexto. Responde a la pregunta del usuario basándote únicamente en el siguiente contexto.
    Si la respuesta no se encuentra en el contexto, di "Lo siento, no tengo información sobre eso en mis documentos". No inventes nada. Sé claro y conciso.
    
    Contexto:
    {context}
    
    Pregunta del usuario:
    {question}<end_of_turn>
    <start_of_turn>model
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        ),  # Aumentamos a 4 fragmentos para el modelo más potente
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    print("\n--- ¡Asistente LOCAL listo! ---")
    print("Escribe tu pregunta o 'salir' para terminar.")

    while True:
        query = input("\nTu pregunta: ")
        if query.lower() == "salir":
            print("¡Hasta la próxima!")
            break
        if query:
            start_time = time.time()
            result = qa_chain.invoke({"query": query})
            end_time = time.time()

            print("\nRespuesta del Experto (GPU):")
            print(result["result"])
            print(f"\n(Respuesta generada en {end_time - start_time:.2f} segundos)")

            print("\nFuentes consultadas:")
            source_files = set(
                os.path.basename(doc.metadata.get("source", "N/A"))
                for doc in result["source_documents"]
            )
            for source_file in source_files:
                print(f"- {source_file}")


if __name__ == "__main__":
    main()
