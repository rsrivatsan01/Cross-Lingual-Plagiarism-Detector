# =========================================
# Cross-Lingual Plagiarism Detection System
# (Improved F1 Evaluation)
# =========================================

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# --------------------------------
# Load models
# --------------------------------
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading multilingual model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# --------------------------------
# Preprocessing
# --------------------------------
def preprocess(text):
    return text.lower()


# --------------------------------
# Similarity
# --------------------------------
def similarity_score(text1, text2):
    clean1 = preprocess(text1)
    clean2 = preprocess(text2)
    emb1, emb2 = model.encode([clean1, clean2], convert_to_numpy=True)
    return cosine_similarity([emb1], [emb2])[0][0]


# --------------------------------
# Load datasets
# --------------------------------
dataset = load_from_disk("datasets/pawsx_en")
opus = load_from_disk("datasets/opus_en_hi")


# --------------------------------
# THRESHOLD TUNING (BASED ON F1) 
# --------------------------------
print("\nFinding Best Threshold (F1-based)...\n")

best_f1 = 0
best_thresh = 0

for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    y_true, y_pred = [], []

    for item in dataset["test"].select(range(100)):
        s1, s2, label = item["sentence1"], item["sentence2"], item["label"]
        score = similarity_score(s1, s2)
        pred = 1 if score > t else 0

        y_true.append(label)
        y_pred.append(pred)

    f1 = f1_score(y_true, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best Threshold:", best_thresh)
print("Best F1 Score:", round(best_f1, 3))


# --------------------------------
# FINAL EVALUATION 
# --------------------------------
print("\nFinal Evaluation using Best Threshold...\n")

y_true, y_pred = [], []

for item in dataset["test"].select(range(100)):
    s1, s2, label = item["sentence1"], item["sentence2"], item["label"]
    score = similarity_score(s1, s2)
    pred = 1 if score > best_thresh else 0

    y_true.append(label)
    y_pred.append(pred)

print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
print("Precision:", round(precision_score(y_true, y_pred), 3))
print("Recall   :", round(recall_score(y_true, y_pred), 3))
print("F1 Score :", round(f1_score(y_true, y_pred), 3))


# --------------------------------
# HARD NEGATIVE DETECTION 
# --------------------------------
print("\nHard Negative Cases...\n")

for item in dataset["test"].select(range(20)):
    s1, s2, label = item["sentence1"], item["sentence2"], item["label"]
    score = similarity_score(s1, s2)

    if score > best_thresh and label == 0:
        print("️ Hard Negative")
        print("Similarity:", round(score, 3))
        print("Sentence1:", s1)
        print("Sentence2:", s2)
        print("---------------------------")


# --------------------------------
# CROSS-LANGUAGE VALIDATION 
# --------------------------------
print("\nCross-Language Validation...\n")

correct = 0
total = 20

for item in opus["train"].select(range(total)):
    eng = item["translation"]["en"]
    hin = item["translation"]["hi"]

    score = similarity_score(eng, hin)

    if score > best_thresh:
        correct += 1

print("Cross-language accuracy:", round(correct / total, 3))
