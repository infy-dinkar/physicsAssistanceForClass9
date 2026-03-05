import streamlit as st
import os
import base64
import tempfile

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import speech_recognition as sr
from gtts import gTTS


# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -----------------------------
# Page Title
# -----------------------------
st.title("Class 9 Physics Assistant")


# -----------------------------
# Load Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Load FAISS Database
# -----------------------------
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------
# Load LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)


# -----------------------------
# Translate Question
# -----------------------------
def translate_to_english(question):

    prompt = f"""
Translate the following question into English.

Question:
{question}

Return ONLY the translated sentence.
"""

    response = llm.invoke(prompt)
    return response.content.strip()


# -----------------------------
# Speech to Text
# -----------------------------
def voice_input():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        st.info("🎤 Listening... Speak your question")

        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text

    except:
        return "Could not understand audio"


# -----------------------------
# Text to Speech (Auto Play)
# -----------------------------
def speak_answer(answer):

    tts = gTTS(answer)

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    tts.save(temp_audio.name)

    audio_file = open(temp_audio.name, "rb")
    audio_bytes = audio_file.read()

    b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)


# -----------------------------
# Input Section
# -----------------------------
query = st.text_input("Ask a Physics Question")

if st.button("Ask with Voice"):

    query = voice_input()
    st.write("You asked:", query)


# -----------------------------
# Main Logic
# -----------------------------
if query:

    # Step 1 → Translate
    english_query = translate_to_english(query)

    # Step 2 → Retrieve documents
    retrieved_docs = vectorstore.similarity_search(english_query, k=3)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])


    # Step 3 → Prompt
    final_prompt = f"""
You are a Physics assistant for Class 9 students.

IMPORTANT RULES:

1. Answer ONLY using the given context.
2. If the question is NOT related to physics say:
"I can only answer physics related questions."
3. If the answer is NOT present in the context say:
"The answer is not available in the provided physics material."
4. Always answer in SIMPLE ENGLISH.

Context:
{context}

Question:
{english_query}

Answer:
"""


    # Step 4 → LLM Answer
    response = llm.invoke(final_prompt)

    answer = response.content


    # Step 5 → Show Answer
    st.subheader("Answer")
    st.write(answer)


    # Step 6 → Speak Answer
    speak_answer(answer)