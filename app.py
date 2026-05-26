"""
app.py
------
This is the Flask backend — the brain of our web application.

Flask is a lightweight Python web framework. It:
  1. Serves the HTML page to the browser.
  2. Accepts POST requests from the frontend (with news text).
  3. Preprocesses the text, runs it through our model, and returns a prediction.

HOW A WEB REQUEST FLOWS:
  Browser  →  POST /predict  →  Flask  →  Model  →  JSON response  →  Browser displays result
"""

import os
import pickle
import string
import time

import nltk
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import fact_checker

# --------------------------------------------------------------------------- #
#  Setup                                                                        #
# --------------------------------------------------------------------------- #

# Check if NLTK data is present before calling download (optimizes boot speed)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

# Initialise Flask. '__name__' tells Flask where to find templates/ and static/
app = Flask(__name__)

# --------------------------------------------------------------------------- #
#  Load the pre-trained model and vectorizer                                    #
# --------------------------------------------------------------------------- #
#
# We load these ONCE at startup. Loading inside the prediction function
# would be slow — reading from disk on every request is expensive.

MODEL_PATH      = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "\n\n  [ERROR] model.pkl or vectorizer.pkl not found!\n"
        "  Please run:  python train.py  first.\n"
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("  [OK] Model and vectorizer loaded successfully.")


# --------------------------------------------------------------------------- #
#  Text preprocessing (must be identical to train.py!)                          #
# --------------------------------------------------------------------------- #
#
# IMPORTANT: The preprocessing here MUST match train.py exactly.
# If they differ, the vectorizer will produce different features and
# predictions will be meaningless.

def preprocess_text(text: str) -> str:
    """
    Clean and normalise user-submitted news text.

    Steps: lowercase → remove punctuation → remove stopwords → stem
    """
    if not isinstance(text, str):
        return ""

    text  = text.lower()
    text  = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in STOP_WORDS]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)


# --------------------------------------------------------------------------- #
#  Routes                                                                       #
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    """
    Serve the main HTML page.

    render_template() looks for index.html inside the /templates folder.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept news text from the frontend, run the model, and return JSON.

    Expected request body (JSON):
      { "text": "Some news headline or article..." }

    Response (JSON):
      {
        "prediction": "FAKE" | "REAL",
        "confidence": 87.3,          <- percentage, e.g. 87.3%
        "fake_prob": 87.3,
        "real_prob": 12.7,
        "word_count": 42,
        "label": "FAKE NEWS"  | "REAL NEWS"
      }
    """
    # --- 1. Extract text from the request ---
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        # Return a 400 Bad Request if no text was sent
        return jsonify({"error": "No text provided."}), 400

    news_text = data["text"].strip()

    if len(news_text) < 10:
        return jsonify({"error": "Text is too short. Please enter more content."}), 400

    # --- 2. Preprocess ---
    cleaned = preprocess_text(news_text)

    if not cleaned:
        return jsonify({"error": "Text could not be processed. Try different content."}), 400

    # --- 3. Vectorize using the SAME TF-IDF vocabulary from training ---
    #
    # transform() (not fit_transform!) — we apply the existing vocabulary,
    # not learn a new one. Any words unseen during training are ignored.
    vector = vectorizer.transform([cleaned])

    # --- 4. ML Predict ---
    #
    # predict()       -> the winning class label: "FAKE" or "REAL"
    # predict_proba() -> probability for each class, e.g. [0.87, 0.13]
    prediction  = model.predict(vector)[0]          # "FAKE" or "REAL"
    proba       = model.predict_proba(vector)[0]    # [prob_class0, prob_class1]

    # model.classes_ is an array like ['FAKE', 'REAL'] — we match indices
    classes     = list(model.classes_)
    ml_fake_prob = round(proba[classes.index("FAKE")] * 100, 1)
    ml_real_prob = round(proba[classes.index("REAL")] * 100, 1)
    ml_confidence = ml_fake_prob if prediction == "FAKE" else ml_real_prob

    word_count  = len(news_text.split())

    # --- 5. Global Verification via web cross-referencing ---
    #
    # This searches DuckDuckGo for the news, checks trusted sources,
    # analyzes language patterns, and combines everything.
    try:
        result = fact_checker.combined_analysis(
            text=news_text,
            ml_prediction=prediction,
            ml_confidence=ml_confidence,
            ml_fake_prob=ml_fake_prob,
            ml_real_prob=ml_real_prob,
        )
    except Exception as e:
        # If web verification fails, fall back to ML-only results
        result = {
            "prediction":    prediction,
            "label":         f"{prediction} NEWS",
            "confidence":    ml_confidence,
            "reason":        f"Web verification unavailable. ML model predicts {prediction} with {ml_confidence}% confidence.",
            "sources":       [],
            "ml_score":      ml_real_prob,
            "web_score":     0,
            "lang_score":    0,
            "fake_prob":     ml_fake_prob,
            "real_prob":     ml_real_prob,
            "trusted_found": 0,
            "total_results": 0,
            "sensational":   False,
        }

    # Add word count to the result
    result["word_count"] = word_count

    # --- 6. Return JSON ---
    return jsonify(result)


# --------------------------------------------------------------------------- #
#  Start the server                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # debug=True gives helpful error pages and auto-reloads on file changes.
    # NEVER use debug=True in production.
    port = int(os.environ.get("PORT", 5000))
    # debug=True is great for local development (auto-reloads on file changes).
    # In production (e.g. Render) set the DEBUG env var to "0".
    debug_mode = os.environ.get("DEBUG", "1") != "0"
    print(f"\n  >>> Starting Flask server on http://localhost:{port}")
    print(f"  Debug mode: {'ON (auto-reload enabled)' if debug_mode else 'OFF'}")
    # use_reloader=False prevents Flask from forking a second process,
    # which confuses some process managers that watch the port.
    app.run(debug=debug_mode, host="0.0.0.0", port=port, use_reloader=False)
