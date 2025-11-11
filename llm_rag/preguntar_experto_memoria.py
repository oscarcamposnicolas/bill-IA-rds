import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate  # Importamos PromptTemplate

# --- CONFIGURACIÓN (sin cambios) ---
VECTORSTORE_NAME = "llm_rag/billar_expert_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL = "gemma3:12b"


def main():
    """
    Función principal para inicializar el chatbot CON MEMORIA y EN ESPAÑOL.
    """
    print(
        f"Iniciando el Asistente de Billar en MODO CONVERSACIONAL... 🧠🇪🇸 (Modelo: {LOCAL_MODEL})"
    )

    # --- Carga de componentes (sin cambios) ---
    print("Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Cargando la base de datos vectorial: '{VECTORSTORE_NAME}'...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_NAME, embeddings, allow_dangerous_deserialization=True
    )

    print(f"Conectando con el modelo local '{LOCAL_MODEL}' vía Ollama...")
    llm = OllamaLLM(
        model=LOCAL_MODEL, temperature=0.2
    )  # Bajamos un poco la temperatura para respuestas más directas

    # --- NUEVO: DEFINIMOS EXPLÍCITAMENTE LOS PROMPTS EN ESPAÑOL ---

    # 1. Prompt para condensar la pregunta (el paso que se nos colaba en inglés)
    condense_question_template = """
    Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, en español.

    Historial del Chat:
    {chat_history}
    
    Pregunta de Seguimiento: {question}
    
    Pregunta Independiente:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    # 2. Prompt para la respuesta final, usando el contexto de los documentos
    qa_template = """
    <start_of_turn>user
    Eres un experto mundial en billar, preciso y riguroso. Tu única fuente de conocimiento son los documentos proporcionados en el contexto. Responde a la pregunta del usuario en español, basándote únicamente en el siguiente contexto.
    Si la respuesta no se encuentra en el contexto, di amablemente "Lo siento, no tengo información sobre eso en mis documentos". No inventes nada. Sé claro y conciso.
    
    Contexto:
    {context}
    
    Pregunta:
    {question}<end_of_turn>
    <start_of_turn>model
    """
    QA_PROMPT = PromptTemplate(
        template=qa_template, input_variables=["context", "question"]
    )

    # --- Memoria (sin cambios) ---
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- Cadena Conversacional (AHORA CON LOS PROMPTS PERSONALIZADOS) ---
    print("Creando la cadena conversacional con prompts en español...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,  # Le pasamos nuestro prompt de condensación
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT
        },  # Le pasamos nuestro prompt de respuesta final
        verbose=False,
    )

    print("\n--- ¡Asistente conversacional listo! (Ahora 100% en español) ---")
    print("Escribe tu pregunta o 'salir' para terminar.")

    # --- Bucle de chat (sin cambios) ---
    while True:
        query = input("\nTu pregunta: ")
        if query.lower() == "salir":
            print("¡Hasta la próxima!")
            break
        if query:
            start_time = time.time()
            result = qa_chain.invoke({"question": query})
            end_time = time.time()

            print("\nRespuesta del Experto (con memoria y en español):")
            print(result["answer"])
            print(f"\n(Respuesta generada en {end_time - start_time:.2f} segundos)")


if __name__ == "__main__":
    main()
