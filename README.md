# 📖 Blog Q&A Chatbot with RAG

A simple yet powerful Streamlit web application that allows you to chat with any blog post or website. This app uses a Retrieval-Augmented Generation (RAG) pipeline to understand the content of a provided URL and answer your questions about it.


---

## 🚀 Features

-   **Web Content Ingestion**: Provide any blog post or website link to start a conversation.
-   **Dynamic Knowledge Base**: Creates a temporary vector store (knowledge base) from the website's content on the fly.
-   **Interactive Chat Interface**: A user-friendly chat interface powered by Streamlit to ask questions.
-   **Powered by Open-Source**: Built with LangChain, Ollama (Gemma 3), ChromaDB, and HuggingFace sentence-transformers.

---

## ⚙️ How It Works

The application follows a standard RAG pipeline to provide answers based on the context of the provided link:

1.  **Load**: The content from the input URL is loaded using `WebBaseLoader`.
2.  **Split**: The loaded document is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
3.  **Embed & Store**: Each chunk is converted into a numerical vector (embedding) using the `all-MiniLM-L6-v2` model from HuggingFace. These embeddings are then stored in an in-memory `Chroma` vector store.
4.  **Retrieve**: When you ask a question, the app embeds your query and retrieves the most relevant text chunks from the vector store based on semantic similarity.
5.  **Generate**: The retrieved chunks (context) and your original question are passed to the `gemma3:1b` model via Ollama. The model then generates a concise answer based on the provided information.

---

## 🛠️ Tech Stack

-   **Web Framework**: [Streamlit](https://streamlit.io/)
-   **LLM Orchestration**: [LangChain](https://www.langchain.com/)
-   **LLM**: [Ollama](https://ollama.com/) with Google's [Gemma 3](https://huggingface.co/google/gemma-2-9b) model
-   **Embedding Model**: [HuggingFace all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
-   **Vector Store**: [ChromaDB](https://www.trychroma.com/) (in-memory)

---

## 📋 Setup and Installation

Follow these steps to run the project on your local machine.

### Prerequisites

-   Python 3.8+
-   [Ollama](https://ollama.com/) installed and running.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/blog-qa-rag.git](https://github.com/your-username/blog-qa-rag.git)
    cd blog-qa-rag
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    Create a file named `requirements.txt` with the following content:
    ```txt
    streamlit
    langchain
    langchain-community
    langchain-chroma
    langchain-huggingface
    python-dotenv
    sentence-transformers
    chromadb
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull the Ollama model:**
    This project uses the `gemma3:1b` model. Pull it from the command line:
    ```bash
    ollama pull gemma3:1b
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Open your web browser and navigate to `http://localhost:8501`.

---

## ▶️ Usage

1.  Launch the application using the command above.
2.  Paste the URL of a blog post or website into the text input field.
3.  Click the "**Start Conversation**" button.
4.  Wait a few moments for the app to process the content and build the knowledge base.
5.  Once you see the "Conversation started!" message, you can ask questions about the content in the chat box at the bottom of the page.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
