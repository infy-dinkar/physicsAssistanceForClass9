# 📘 Physics RAG Chatbot  
### Streamlit + FAISS + LangChain + Groq

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, FAISS, LangChain, HuggingFace Embeddings, and Groq LLM.

This project allows users to ask Physics-related questions and get context-aware answers generated from a PDF knowledge base.

---

## 🚀 Features

- 📄 PDF-based knowledge retrieval
- 🔍 FAISS vector similarity search
- 🤖 Groq LLM for answer generation
- 🌐 Streamlit web interface
- 🌍 English response generation
- 🔒 Secure API key handling using `.env`

---

## 🧠 How It Works (RAG Pipeline)

1. User enters a question in the web interface.
2. The question is converted into embeddings.
3. FAISS searches the most similar document chunks.
4. Retrieved context + question is sent to Groq LLM.
5. LLM generates a clear answer in English.
6. Answer is displayed on the Streamlit webpage.

---

## 📂 Project Structure
1STRAGPROJECT/
│
├── data/ # PDF files
├── faiss_index/ # Saved FAISS vector database
├── venv/ # Virtual environment (ignored in Git)
├── .env # API key file (ignored in Git)
├── rag.ipynb # Notebook for building RAG system
├── app.py # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md


---

## 📦 Dependencies

- streamlit
- langchain
- langchain-community
- langchain-groq
- faiss-cpu
- sentence-transformers
- python-dotenv

---

## 🔒 Security Notes

- `.env` is excluded using `.gitignore`
- `venv/` is excluded
- `faiss_index/` is excluded
- API keys are never pushed to GitHub

---

## 🎯 Future Improvements

- Chat history memory
- Language toggle (English / Hindi)
- Upload your own PDF feature
- Cloud deployment (Render / HuggingFace Spaces)
- UI improvements

---

## 👨‍💻 Author

Built as a learning project to understand:

- Retrieval-Augmented Generation (RAG)
- Vector Databases (FAISS)
- LLM Integration
- Full AI Web App Development