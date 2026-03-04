import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page title
st.title("📘CLASS-9 PHYSICS ASSISTANCE")

# Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

# User input box
query = st.text_input("Ask Physics Question")

if query:
    # Retrieve similar chunks
    retrieved_docs = vectorstore.similarity_search(query, k=3)

    context = ""
    for doc in retrieved_docs:
        context = context + "\n\n" + doc.page_content

    # Prompt
    final_prompt = f"""
    You are a physics teacher.

    Context:
    {context}

    Question:
    {query}

    Answer in simple english.
    """

    response = llm.invoke(final_prompt)

    st.subheader("Answer:")
    st.write(response.content)