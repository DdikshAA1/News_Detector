"""
train.py
--------
This script handles the ENTIRE machine learning pipeline:
  1. Load (or generate) the dataset
  2. Preprocess text (clean, tokenize, stem)
  3. Vectorize text using TF-IDF
  4. Train a Logistic Regression classifier
  5. Evaluate accuracy on a held-out test set
  6. Save the trained model and vectorizer to disk

Run this file ONCE before starting the Flask app:
  $ python train.py

HOW THE ML PIPELINE WORKS (beginner explanation):
  Raw Text --> Preprocessing --> TF-IDF Vectors --> Logistic Regression --> FAKE / REAL
"""

import os
import pickle
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------- #
#  CONFIGURATION                                                                #
# --------------------------------------------------------------------------- #

# Set USE_REAL_DATASET = True if you have downloaded the Kaggle ISOT dataset
# and placed Fake.csv and True.csv inside the /data folder.
USE_REAL_DATASET = False

FAKE_CSV   = "data/Fake.csv"   # Kaggle ISOT fake news file
TRUE_CSV   = "data/True.csv"   # Kaggle ISOT real news file
SYNTH_CSV  = "data/news_dataset.csv"  # auto-generated synthetic data

MODEL_PATH      = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

# --------------------------------------------------------------------------- #
#  STEP 1 — Download NLTK resources                                             #
# --------------------------------------------------------------------------- #

print("=" * 60)
print("  FAKE NEWS DETECTOR — MODEL TRAINING")
print("=" * 60)
print("\n[1/6] Downloading NLTK resources...")

# 'stopwords' is a list of common English words (the, is, at …)
# that carry little meaning and should be removed during preprocessing.
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()


# --------------------------------------------------------------------------- #
#  STEP 2 — Text preprocessing                                                  #
# --------------------------------------------------------------------------- #

def preprocess_text(text: str) -> str:
    """
    Clean and normalise a piece of text so the model can learn from it.

    Steps:
      1. Lowercase   — 'Breaking' and 'breaking' mean the same thing.
      2. Remove punctuation — commas, exclamation marks etc. add noise.
      3. Remove stopwords  — 'the', 'is', 'at' don't help classification.
      4. Stemming          — 'running', 'runs', 'ran' all become 'run'.

    Parameters:
        text : raw news headline or article text

    Returns:
        A single cleaned string ready for TF-IDF vectorization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    #    str.maketrans creates a mapping that replaces every punctuation char with None.
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Split into words, remove stopwords
    words = [w for w in text.split() if w not in STOP_WORDS]

    # 4. Stem each word (reduces a word to its root form)
    words = [stemmer.stem(w) for w in words]

    return " ".join(words)


# --------------------------------------------------------------------------- #
#  STEP 3 — Load dataset                                                        #
# --------------------------------------------------------------------------- #

print("\n[2/6] Loading dataset...")

if USE_REAL_DATASET and os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV):
    # --- Real ISOT Kaggle dataset ---
    print("  Using real ISOT dataset from Kaggle.")
    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    # Both files have a 'title' and 'text' column.
    # We combine them for richer features.
    fake_df["content"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
    true_df["content"] = true_df["title"].fillna("") + " " + true_df["text"].fillna("")

    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    df = pd.concat([fake_df[["content", "label"]],
                    true_df[["content", "label"]]], ignore_index=True)
    text_column = "content"

else:
    # --- Synthetic dataset ---
    if not os.path.exists(SYNTH_CSV):
        print("  Generating synthetic dataset (no real dataset found)...")
        from data.generate_dataset import create_dataset
        create_dataset(n_samples=3000, output_path=SYNTH_CSV)
    else:
        print("  Using existing synthetic dataset.")

    df = pd.read_csv(SYNTH_CSV)
    text_column = "text"

print(f"  Loaded {len(df)} rows.")
print(f"  Label distribution:\n{df['label'].value_counts()}")


# --------------------------------------------------------------------------- #
#  STEP 4 — Preprocess all texts                                                #
# --------------------------------------------------------------------------- #

print("\n[3/6] Preprocessing text (this may take a moment)...")

df["cleaned"] = df[text_column].apply(preprocess_text)

# Drop any rows where cleaning produced an empty string
df = df[df["cleaned"].str.strip() != ""].reset_index(drop=True)
print(f"  {len(df)} rows remain after cleaning.")


# --------------------------------------------------------------------------- #
#  STEP 5 — TF-IDF Vectorization                                                #
# --------------------------------------------------------------------------- #
#
# TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numbers.
#
# TF  = how often a word appears in THIS document
# IDF = how rare the word is across ALL documents
#
# Words that appear a lot in one article but rarely elsewhere get a HIGH score.
# Common words that appear everywhere get a LOW score.
#
# max_features=10000 means we only keep the 10,000 most important words.
# ngram_range=(1,2) means we also consider pairs of consecutive words
#   e.g. "not good" is kept as one feature, not split into "not" and "good".

print("\n[4/6] Vectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=10_000,    # vocabulary size cap
    ngram_range=(1, 2),     # unigrams AND bigrams
    sublinear_tf=True,      # apply log(TF) to reduce the impact of very common words
)

X = vectorizer.fit_transform(df["cleaned"])  # returns a sparse matrix
y = df["label"]                              # FAKE or REAL

print(f"  Feature matrix shape: {X.shape}")


# --------------------------------------------------------------------------- #
#  STEP 6 — Train / Test Split                                                  #
# --------------------------------------------------------------------------- #
#
# We split the data into:
#   80% training   — the model LEARNS from these
#   20% testing    — we EVALUATE on these (model never saw them during training)
#
# stratify=y ensures both splits have the same proportion of FAKE and REAL.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print(f"\n  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")


# --------------------------------------------------------------------------- #
#  STEP 7 — Train Logistic Regression                                           #
# --------------------------------------------------------------------------- #
#
# Logistic Regression is a simple but powerful classification algorithm.
# Despite its name it is used for CLASSIFICATION (not regression).
#
# It learns a weight for each TF-IDF feature. A word like "SHOCKING" gets a
# high weight toward FAKE, while "announced" gets a high weight toward REAL.
#
# max_iter=1000  — allow enough iterations for the optimiser to converge.
# C=1.0          — regularisation strength; smaller = stronger regularisation.

print("\n[5/6] Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
model.fit(X_train, y_train)


# --------------------------------------------------------------------------- #
#  STEP 8 — Evaluate the model                                                  #
# --------------------------------------------------------------------------- #

print("\n[6/6] Evaluating model performance...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  [OK] Accuracy : {accuracy * 100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred))

print("  Confusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
print("  [FAKE,FAKE]  [FAKE,REAL]")
print("  [REAL,FAKE]  [REAL,REAL]")


# --------------------------------------------------------------------------- #
#  STEP 9 — Save model and vectorizer                                           #
# --------------------------------------------------------------------------- #
#
# pickle serialises Python objects to a binary file so they can be loaded
# later without retraining. We save TWO things:
#   model.pkl      — the trained Logistic Regression weights
#   vectorizer.pkl — the fitted TF-IDF vocabulary (MUST match the model!)

os.makedirs("model", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\n  [SAVED] Model saved      -> {MODEL_PATH}")
print(f"  [SAVED] Vectorizer saved -> {VECTORIZER_PATH}")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE — you can now run: python app.py")
print("=" * 60)
