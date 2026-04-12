# =========================================
# Fine-Tuning Sentence-BERT on PAWS-X
# =========================================

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


# --------------------------------
# Load local PAWS-X dataset
# --------------------------------
print("Loading PAWS-X dataset...")

dataset = load_from_disk("datasets/pawsx_en")

# use small subset for fast training
train_data = dataset["train"].select(range(1000))

print("Dataset loaded!")


# --------------------------------
# Convert to training format
# --------------------------------
print("Preparing training data...")

train_examples = []

for item in train_data:
    s1 = item["sentence1"]
    s2 = item["sentence2"]
    label = float(item["label"])  # 1 or 0

    train_examples.append(InputExample(texts=[s1, s2], label=label))

print(f"Total training samples: {len(train_examples)}")


# --------------------------------
# Load pretrained model
# --------------------------------
print("Loading pretrained model...")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# --------------------------------
# Training setup
# --------------------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)


# --------------------------------
# Train model
# --------------------------------
print("\nTraining started...\n")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,           # keep low for demo
    warmup_steps=100,
    show_progress_bar=True
)


# --------------------------------
# Save trained model
# --------------------------------
model.save("fine_tuned_model")

print("\nTraining complete!")
print("Model saved as: fine_tuned_model")
