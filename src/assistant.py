import json
import os
import time

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Define the path for the persistent FAISS index
FAISS_INDEX_PATH = "faiss_index"

def load_documents(file_path='project_1_publications.json'):
    """
    Loads and processes documents from the specified JSON file.
    This follows the principle of reading your files first.
    """
    print("Loading documents...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None

    documents = []
    for item in data:
        content = item.get("publication_description", "")
        metadata = {"title": item.get("title", "No Title")}
        documents.append(Document(page_content=content, metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Loaded and split {len(docs)} document chunks.")
    return docs

def create_vector_store(docs, embeddings):
    """
    Creates and saves a FAISS vector store from document chunks and their embeddings.
    This pushes the vectors and text to the database.
    """
    print("Creating and saving FAISS vector store...")
    vector = FAISS.from_documents(docs, embeddings)
    vector.save_local(FAISS_INDEX_PATH)
    print(f"Vector store saved to '{FAISS_INDEX_PATH}'.")
    return vector

def load_vector_store(embeddings):
    """
    Loads an existing FAISS vector store from the local disk.
    """
    print(f"Loading FAISS vector store from '{FAISS_INDEX_PATH}'...")
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def create_rag_chain(llm, retriever):
    """
    Creates the RAG (Retrieval-Augmented Generation) chain.
    The goal is to pass the answers from the vector database to your LLM.
    """
    prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        Your answer must be based *exclusively* on the provided context. Do not use any other information.
        If you don't know the answer from the context, just say that you don't know. Do not try to make up an answer.

        <context>
        {context}
        </context>
        Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def main():
    """
    Main function to run the RAG assistant with performance tracking.
    """
    # Initialize Ollama embeddings using the Mistral model
    ollama_embeddings = OllamaEmbeddings(model="mistral")

    # --- Indexing Time Measurement ---
    # Check if the vector store already exists, otherwise create it
    if os.path.exists(FAISS_INDEX_PATH):
        start_time = time.perf_counter()
        vector_store = load_vector_store(ollama_embeddings)
        end_time = time.perf_counter()
        print(f"Vector store loaded in {end_time - start_time:.2f} seconds.")
    else:
        docs = load_documents()
        if not docs:
            return
        start_time = time.perf_counter()
        vector_store = create_vector_store(docs, ollama_embeddings)
        end_time = time.perf_counter()
        print(f"Vector store created and saved in {end_time - start_time:.2f} seconds.")

    # Initialize the Ollama LLM with the Mistral model
    llm = Ollama(model="mistral")

    retrieval_chain = create_rag_chain(llm, vector_store.as_retriever())
    print("\nRAG chain created successfully.")
    print("\nReady to Answer Questions! (Type 'exit' to quit)")

    # --- Query Time Measurement ---
    while True:
        try:
            user_input = input("\nYour Question: ")
            if user_input.lower() == 'exit':
                print("Exiting assistant. Goodbye!")
                break

            start_query_time = time.perf_counter()
            response = retrieval_chain.invoke({"input": user_input})
            end_query_time = time.perf_counter()

            print("\nAssistant's Answer:")
            print(response["answer"])
            print(f"\n(Responded in {end_query_time - start_query_time:.2f} seconds)")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()