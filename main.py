# =========================================
# Cross-Lingual Plagiarism Detection System
# =========================================

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score


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
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)


# --------------------------------
# Similarity
# --------------------------------
def similarity_score(text1, text2):
    clean1 = preprocess(text1)
    clean2 = preprocess(text2)
    emb1, emb2 = model.encode([clean1, clean2], convert_to_numpy=True)
    return cosine_similarity([emb1], [emb2])[0][0]


# --------------------------------
# Classification
# --------------------------------
def classify(score):
    if score >= 0.85:
        return "Direct Plagiarism"
    elif score >= 0.65:
        return "Possible Paraphrased Plagiarism"
    else:
        return "Different Meaning"


# --------------------------------
# Explanation
# --------------------------------
def explain(text1, text2, score):
    if score > 0.75:
        return "Both sentences convey similar meaning with different wording."
    elif score > 0.5:
        return "Sentences share partial semantic similarity."
    else:
        return "Sentences have different meanings."


# --------------------------------
# Colored Output
# --------------------------------
def print_colored(label):
    if label == "Direct Plagiarism":
        print("\033[91m" + label + "\033[0m")
    elif label == "Possible Paraphrased Plagiarism":
        print("\033[93m" + label + "\033[0m")
    else:
        print("\033[92m" + label + "\033[0m")


# --------------------------------
# Load datasets
# --------------------------------
dataset = load_from_disk("datasets/pawsx_en")
opus = load_from_disk("datasets/opus_en_hi")


# --------------------------------
# EXAMPLES
# --------------------------------
print("\nRunning Examples...\n")

examples = [
    ("Artificial intelligence is transforming modern technology.",
     "कृत्रिम बुद्धिमत्ता आधुनिक तकनीक को बदल रही है।"),
    ("Machine learning helps computers learn from data.",
     "El aprendizaje automático permite que las computadoras aprendan de los datos."),
    ("Deep learning improves image recognition systems.",
     "Neural networks help machines recognize images more accurately.")
]

for i, (t1, t2) in enumerate(examples):
    score = similarity_score(t1, t2)
    label = classify(score)

    print(f"Example {i+1}")
    print("Similarity:", round(score, 3))
    print("Result:", end=" ")
    print_colored(label)
    print("Explanation:", explain(t1, t2, score))
    print("---------------------------")


# --------------------------------
# PAWS-X Evaluation
# --------------------------------
print("\nPAWS-X Evaluation...\n")

y_true, y_pred = [], []

for item in dataset["test"].select(range(50)):
    s1, s2, label = item["sentence1"], item["sentence2"], item["label"]
    score = similarity_score(s1, s2)
    pred = 1 if score > 0.7 else 0

    y_true.append(label)
    y_pred.append(pred)

print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
print("F1 Score:", round(f1_score(y_true, y_pred), 3))


# --------------------------------
# HARD NEGATIVE DETECTION 🔥
# --------------------------------
print("\nHard Negative Cases...\n")

for item in dataset["test"].select(range(20)):
    s1, s2, label = item["sentence1"], item["sentence2"], item["label"]
    score = similarity_score(s1, s2)

    if score > 0.7 and label == 0:
        print("⚠️ Hard Negative")
        print("Similarity:", round(score, 3))
        print("Sentence1:", s1)
        print("Sentence2:", s2)
        print("---------------------------")


# --------------------------------
# THRESHOLD TUNING 🔥
# --------------------------------
print("\nFinding Best Threshold...\n")

best_acc = 0
best_thresh = 0

for t in [0.5, 0.6, 0.7, 0.8]:
    y_true, y_pred = [], []

    for item in dataset["test"].select(range(50)):
        s1, s2, label = item["sentence1"], item["sentence2"], item["label"]
        score = similarity_score(s1, s2)
        pred = 1 if score > t else 0

        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)

    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("Best Threshold:", best_thresh)
print("Best Accuracy:", round(best_acc, 3))


# --------------------------------
# DATASET STATS 🔥
# --------------------------------
print("\nDataset Stats...\n")

labels = [item["label"] for item in dataset["train"].select(range(200))]

print("Paraphrase:", sum(labels))
print("Non-Paraphrase:", len(labels) - sum(labels))


# --------------------------------
# CROSS-LANGUAGE VALIDATION 🔥
# --------------------------------
print("\nCross-Language Validation...\n")

correct = 0
total = 20

for item in opus["train"].select(range(total)):
    eng = item["translation"]["en"]
    hin = item["translation"]["hi"]

    score = similarity_score(eng, hin)

    if score > 0.6:
        correct += 1

print("Cross-language accuracy:", round(correct / total, 3))
