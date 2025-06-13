import json
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings

from langchain_ollama import OllamaEmbeddings

from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

print("Step 1: Loading and Processing Documents...")
try:
    with open('project_1_publications.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: 'project_1_publications.json' not found.")
    print("Please make sure the JSON file is in the same directory as this script.")
    exit()

documents = []

for item in data:
    content = item.get("publication_description", "")
    metadata = {"title": item.get("title", "No Title")}
    documents.append(Document(page_content=content, metadata=metadata))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print(f"Loaded and split {len(docs)} document chunks.")

print("\nStep 2: Creating Embeddings and Storing in FAISS...")
ollama_embeddings = OllamaEmbeddings(model="mistral")

vector = FAISS.from_documents(docs, ollama_embeddings)
print("Embeddings created and stored in FAISS.")

print("\nStep 3: Initializing LLM and Creating RAG Chain...")
# Initialize the Ollama LLM with the Mistral model
llm = Ollama(model="mistral")


prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step.
If you don't know the answer, just say that you don't know.
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)


retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("RAG chain created successfully.")


print("\nStep 4: Ready to Answer Questions!")
print("Type 'exit' to quit.")

while True:
    try:
        user_input = input("\nYour Question: ")
        if user_input.lower() == 'exit':
            print("Exiting assistant. Goodbye!")
            break

        # Invoke the chain with the user's question
        response = retrieval_chain.invoke({"input": user_input})

        # Print the answer
        print("\nAssistant's Answer:")
        print(response["answer"])

    except Exception as e:
        print(f"An error occurred: {e}")