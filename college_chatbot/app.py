import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import traceback
import base64


# -------------------------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Gujarat Vidyapith Chatbot",
    layout="centered"
)


# -------------------------------------------------
# FUNCTION: SET BACKGROUND + OPACITY LAYER
# -------------------------------------------------
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Overlay for opacity */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.45);
            z-index: -1;
        }}

        /* Answer Card Styling */
        .answer-box {{
            background: rgba(255, 255, 255, 0.92);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.25);
            font-size: 16px;
            line-height: 1.5;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Set background
set_bg("bg2.jpg")


# -------------------------------------------------
# FIX WARNING BOX COLOR (YELLOW → WHITE)
# -------------------------------------------------
st.markdown("""
<style>
/* Make Streamlit warning box white */
.stAlert {
    background-color: white !important;
    border-left: 4px solid orange !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# TITLE & HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: white; text-shadow: 0px 0px 6px black;'>
        Campus Saathi
    </h1>
    <h4 style='text-align:center; color:white; margin-top:-10px; text-shadow:0px 0px 4px black;'>
        Ask anything about admissions, courses, campus & more
    </h4>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# LOAD MODEL & DATABASE
# -------------------------------------------------
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "college"


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


model = load_model()
collection = load_collection()


# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
query = st.text_input(
    "💬 Ask your question",
    placeholder="Example: What courses are offered?"
)


# -------------------------------------------------
# GREETING DETECTION
# -------------------------------------------------
greetings = ["hi", "hello", "hey", "namaste", "good morning", "good evening", "good afternoon"]


def is_greeting(q):
    return any(g in q.lower() for g in greetings)


# -------------------------------------------------
# ASK BUTTON ACTION
# -------------------------------------------------
if st.button("Ask"):

    if not query.strip():
        st.warning("⚠ Please enter a question first.")

    else:

        # Greeting Shortcut
        if is_greeting(query):
            st.markdown(
                "<div class='answer-box'>🙏 Namaste! How can I assist you regarding Gujarat Vidyapith?</div>",
                unsafe_allow_html=True
            )

        else:
            try:
                emb = model.encode([query]).tolist()
                result = collection.query(query_embeddings=emb, n_results=3)
                docs = result.get("documents", [[]])[0]

                if docs:
                    answer = docs[0][:500]  # Trim long text

                    st.markdown(
                        f"<div class='answer-box'>{answer}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.error("❌ No related answer found.")

            except Exception:
                st.error("❌ Something went wrong.")
                st.code(traceback.format_exc())
