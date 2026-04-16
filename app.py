import streamlit as st
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==============================
# Model Loading
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ==============================
# Dataset Loading
# ==============================
@st.cache_data
def load_data():
    return load_from_disk("datasets/pawsx_en")

model = load_model()

# ==============================
# Text Preprocessing
# ==============================
def preprocess(text):
    return text.strip()

# ==============================
# Similarity Computation
# ==============================
def similarity_score(text1, text2):
    emb = model.encode(
        [preprocess(text1), preprocess(text2)],
        normalize_embeddings=True
    )
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

# ==============================
# Classification Logic
# ==============================
def classify(score):
    if score >= 0.9:
        return "Direct Plagiarism"
    elif score >= 0.75:
        return "Possible Paraphrased Plagiarism"
    else:
        return "Different Meaning"

# ==============================
# UI - Input Section
# ==============================
st.title("Cross-Lingual Plagiarism Detection System")

st.write("Enter two texts in any language to evaluate semantic similarity.")

text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

# ==============================
# Similarity Result Section
# ==============================
if st.button("Check Similarity"):
    if text1 and text2:
        score = similarity_score(text1, text2)
        label = classify(score)

        st.subheader("Result")
        st.write(f"Similarity Score: {round(score, 3)}")
        st.write(f"Classification: {label}")

        # Explanation of score
        st.markdown("### Interpretation")
        if score >= 0.9:
            st.success("The texts are highly similar and likely directly plagiarized.")
        elif score >= 0.75:
            st.warning("The texts share strong semantic similarity, possibly paraphrased.")
        elif score >= 0.5:
            st.info("There is some overlap in meaning but not strong enough for plagiarism.")
        else:
            st.write("The texts are semantically different.")

    else:
        st.error("Please enter both texts.")

# ==============================
# Evaluation Section
# ==============================
st.header("Model Evaluation")

if st.button("Run Evaluation"):
    dataset = load_data()

    # Random sampling for unbiased evaluation
    indices = list(range(len(dataset["test"])))
    random.shuffle(indices)
    sample_indices = indices[:300]

    subset = dataset["test"].select(sample_indices)

    sentences1 = [x["sentence1"] for x in subset]
    sentences2 = [x["sentence2"] for x in subset]
    y_true = [x["label"] for x in subset]

    # Batch embedding for efficiency
    emb1 = model.encode(sentences1, batch_size=32, normalize_embeddings=True)
    emb2 = model.encode(sentences2, batch_size=32, normalize_embeddings=True)

    scores = cosine_similarity(emb1, emb2).diagonal()

    st.markdown("### Threshold Tuning (F1-based)")

    best_f1 = 0
    best_thresh = 0

    for t in [i / 100 for i in range(70, 91, 5)]:
        preds = [1 if s > t else 0 for s in scores]
        f1 = f1_score(y_true, preds)

        st.write(f"Threshold {t:.2f} → F1 Score: {round(f1, 3)}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    st.write(f"Best Threshold: {best_thresh}")
    st.write(f"Best F1 Score: {round(best_f1, 3)}")

    # Final evaluation using best threshold
    preds = [1 if s > best_thresh else 0 for s in scores]

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    st.subheader("Final Metrics")

    st.write(f"Accuracy : {round(acc, 3)}")
    st.write(f"Precision: {round(prec, 3)}")
    st.write(f"Recall   : {round(rec, 3)}")
    st.write(f"F1 Score : {round(f1, 3)}")

    # ==============================
    # Metrics Explanation
    # ==============================
    st.markdown("### Metric Explanation")

    st.markdown("""
- **Accuracy**: Overall correctness of predictions (both plagiarism and non-plagiarism).

- **Precision**: Out of all predicted plagiarism cases, how many were actually correct.
  - High precision → fewer false positives.

- **Recall**: Out of all actual plagiarism cases, how many were detected.
  - High recall → fewer missed plagiarism cases.

- **F1 Score**: Harmonic mean of precision and recall.
  - Provides a balanced measure when classes are uneven.

- **Threshold**: The similarity cut-off used to classify plagiarism.
  - Lower threshold → more sensitive (higher recall)
  - Higher threshold → more strict (higher precision)
""")
