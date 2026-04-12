import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Load models (only once)
# -------------------------------
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return nlp, model

nlp, model = load_models()


# -------------------------------
# Functions
# -------------------------------
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and not t.is_punct])


def similarity_score(t1, t2):
    e1, e2 = model.encode([preprocess(t1), preprocess(t2)])
    return cosine_similarity([e1], [e2])[0][0]


def classify(score):
    if score >= 0.85:
        return "🔴 Direct Plagiarism"
    elif score >= 0.65:
        return "🟡 Paraphrased"
    else:
        return "🟢 Different"


# -------------------------------
# UI (IMPORTANT)
# -------------------------------
st.title("🔍 Cross-Lingual Plagiarism Detector")

st.write("Enter two texts to check similarity")

text1 = st.text_area("Text 1")
text2 = st.text_area("Text 2")

if st.button("Check Similarity"):

    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts")
    else:
        score = similarity_score(text1, text2)
        result = classify(score)

        st.success(f"Similarity Score: {round(score, 3)}")
        st.write("### Result:", result)
