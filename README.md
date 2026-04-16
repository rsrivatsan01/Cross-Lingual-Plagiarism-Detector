# Cross-Lingual Plagiarism Detection using Fine-Tuned Sentence-BERT
## Overview
# Designed by R Srivatsan (23BCE0004)

This project presents an advanced cross-lingual plagiarism detection system built using modern NLP techniques.
Unlike traditional plagiarism detectors that rely on surface-level word matching, this system focuses on deep semantic understanding. It is capable of identifying plagiarism even when the text is heavily paraphrased or written in a different language.
To achieve this, a multilingual Sentence-BERT model is fine-tuned on the PAWS-X dataset, which is specifically designed for challenging paraphrase detection tasks.

## Problem Statement

Detecting plagiarism becomes significantly harder when:
- Words are replaced with synonyms
- Sentence structure is altered
- Content is translated into another language

This project addresses these challenges by leveraging semantic similarity instead of lexical similarity.

## Key Innovations

- Cross-lingual capability using a shared embedding space  
- Paraphrase-aware detection based on meaning rather than wording  
- Fine-tuned transformer model for improved performance  
- F1-score-based threshold optimization  
- Efficient batch processing for faster evaluation  
- Interactive Streamlit interface  

## Model Architecture

- Base Model: `paraphrase-multilingual-MiniLM-L12-v2`  
- Framework: Sentence-BERT  
- Fine-Tuning Dataset: PAWS-X  
- Training Objective: Cosine Similarity Loss  

The model converts sentences into dense vector representations (embeddings), enabling comparison based on meaning.

## Fine-Tuning Strategy

The model is trained on sentence pairs:

- Input: (sentence1, sentence2)  
- Label:
  - 1 → Same meaning (paraphrase)  
  - 0 → Different meaning  

### Training Pipeline

1. Load PAWS-X dataset  
2. Select representative training subset  
3. Convert data into InputExample format  
4. Train using cosine similarity loss  
5. Save fine-tuned model  

### Run Training

bash
python train_model.py
