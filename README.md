# Local RAG Question-Answering Assistant

This project is a simple, yet powerful, question-answering assistant that runs entirely on your local machine. It uses a Retrieval-Augmented Generation (RAG) architecture to answer questions based on a custom knowledge base of AI and Machine Learning publications.

The core of this project is built with **LangChain** and leverages a local **Mistral** model served by **Ollama**, with a **FAISS** vector store for efficient document retrieval.

## How It Works

The assistant operates on the RAG model, which grounds the language model's responses in factual data, reducing inaccuracies and fabrications. The process is as follows:

1.  **Load & Chunk**: The system first ingests the knowledge base (`project_1_publications.json`) and splits the text into smaller, overlapping chunks.
2.  **Embed & Store**: Each chunk is converted into a numerical vector (embedding) using the local Mistral model. These vectors are then stored in a local FAISS database for fast retrieval.
3.  **Retrieve & Generate**: When a user asks a question, the system finds the most relevant text chunks from the FAISS database and passes them, along with the original question, to the Mistral model. The model then generates a comprehensive answer based on the provided context.

## Prerequisites

Before you begin, ensure you have the following installed and set up:

* **Python 3.8+**
* **Ollama**: Follow the installation instructions at [ollama.com](https://ollama.com/).
* **Mistral Model**: Pull the Mistral model through Ollama by running the following command in your terminal:
    ```bash
    ollama pull mistral
    ```

## Installation

To get the assistant set up and running, follow these steps:

1.  **Clone the Repository** 
    ```bash
    git clone https://github.com/gourishankar27/local_RAG_agent_system.git
    cd local_RAG_agent_system
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Required Dependencies**
    This project's dependencies are listed in the `requirements.txt` file. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the installation is complete, you can start the assistant.

1.  **Run the application** from your terminal:
    ```bash
    python assistant.py
    ```

2.  **Wait for the setup to complete**. The script will first process the documents and build the FAISS vector store. You will see progress messages for each step.

3.  **Ask a question**. Once you see the `Your Question:` prompt, you can start asking questions.

    **Sample Questions:**
    * `What are the core components of an AI agent?`
    * `Can you explain what QLoRA is and why it's efficient?`
    * `How do you build a RAG application?`

4.  **Exit the application** by typing `exit` and pressing Enter.


## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

