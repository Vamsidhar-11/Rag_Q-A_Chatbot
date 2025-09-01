import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize LLM & embeddings once
llm = Ollama(model="gemma3:1b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---- Function to process blog and build chain ----
def build_rag_chain(link: str):
    loader = WebBaseLoader(web_paths=(link,))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep it concise (max 3 sentences)."
        "\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# ---- Streamlit App ----
st.set_page_config(page_title="RAG Blog Q&A", page_icon="📖")

st.title("📖 Blog Q&A Chatbot")

# Step 1: Input blog link
blog_link = st.text_input("Enter a blog/website link:")

if st.button("Start Conversation"):
    if blog_link:
        with st.spinner("Processing blog and creating knowledge base..."):
            st.session_state.rag_chain = build_rag_chain(blog_link)
            st.session_state.messages = []  # reset chat history
        st.success("✅ Conversation started! Ask your questions below.")
    else:
        st.warning("⚠️ Please enter a valid blog link.")


# Step 2: Q&A chatbot (only after link is processed)
if "rag_chain" in st.session_state:
    st.subheader("💬 Chat with Blog")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if query := st.chat_input("Ask a question about the blog..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({"input": query})
                answer = response.get("answer") or response.get("output_text", "No answer found.")
                st.write(answer)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
