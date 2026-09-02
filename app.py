"""
Part 4 — Flask Web Interface
=============================
A single-page web app that ties together the document assistant (Part 1)
and the lead scorer (Part 3). A salesperson can:
  - Ask the document assistant a question and see the answer with its source
  - Enter a lead's details and see its conversion probability score

Run:
  python app.py
  Then open http://127.0.0.1:5000 in your browser.
"""

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, jsonify

# Import the document assistant from Part 1
from part1_document_assistant import (
    build_knowledge_base,
    answer_question,
)


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "lead_scoring_model.pkl"

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Load Part 1 knowledge base (TF-IDF index) at startup
# ---------------------------------------------------------------------------
print("Loading document knowledge base...")
chunk_texts, chunk_labels, vectorizer, tfidf_matrix = build_knowledge_base()
print("Document assistant ready.")

# ---------------------------------------------------------------------------
# Load Part 3 model at startup
# ---------------------------------------------------------------------------
print("Loading lead scoring model...")
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)
    lead_model = model_artifact["model"]
    label_encoders = model_artifact["label_encoders"]
    feature_order = model_artifact["feature_order"]
    print("Lead scoring model ready.")
else:
    lead_model = None
    print("WARNING: lead_scoring_model.pkl not found. Run part3_lead_scoring.py first.")


# ---------------------------------------------------------------------------
# City normalization (same as Part 3, kept here for the web form input)
# ---------------------------------------------------------------------------
CITY_NORMALIZATION = {
    "islamabad": "Islamabad", "isb": "Islamabad",
    "rawalpindi": "Rawalpindi", "rwp": "Rawalpindi",
    "lahore": "Lahore", "karachi": "Karachi", "khi": "Karachi",
    "peshawar": "Peshawar", "faisalabad": "Faisalabad",
    "multan": "Multan", "gujranwala": "Gujranwala",
    "abbottabad": "Abbottabad",
}


def normalize_city(city_val):
    """Map messy city names to a canonical form."""
    cleaned = str(city_val).strip().lower()
    return CITY_NORMALIZATION.get(cleaned, city_val.strip().title())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_document():
    """Handle a question for the document assistant."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    answer = answer_question(
        question, vectorizer, tfidf_matrix, chunk_texts, chunk_labels
    )
    return jsonify({"answer": answer})


@app.route("/score", methods=["POST"])
def score_lead():
    """Score a lead's conversion probability."""
    if lead_model is None:
        return jsonify({
            "error": "Model not loaded. Run part3_lead_scoring.py first."
        }), 500

    data = request.get_json()

    try:
        # Build a feature row matching the model's expected input
        city_normalized = normalize_city(data.get("city", "Islamabad"))

        raw_features = {
            "source": data.get("source", "Facebook Ads"),
            "city": city_normalized,
            "area": data.get("area", "Unknown"),
            "property_type": data.get("property_type", "Apartment"),
            "budget_pkr_lac": float(data.get("budget_pkr_lac", 100)),
            "bedrooms": int(data.get("bedrooms", 2)),
            "first_response_minutes": float(data.get("first_response_minutes", 30)),
            "calls_made": int(data.get("calls_made", 1)),
            "total_call_seconds": float(data.get("total_call_seconds", 120)),
            "whatsapp_replies": int(data.get("whatsapp_replies", 2)),
            "site_visits": int(data.get("site_visits", 0)),
            "agent_experience_years": float(data.get("agent_experience_years", 2)),
            "is_overseas": int(data.get("is_overseas", 0)),
            "referred_by_existing_client": int(data.get("referred_by_existing_client", 0)),
            "has_financing_approved": int(data.get("has_financing_approved", 0)),
        }

        # Encode categoricals using the same LabelEncoders from training
        for col in ["source", "city", "area", "property_type"]:
            le = label_encoders[col]
            val = raw_features[col]
            if val in le.classes_:
                raw_features[col] = le.transform([val])[0]
            else:
                # Unseen category — fall back to the most common class
                raw_features[col] = 0

        # Build the feature vector in the exact order the model expects
        feature_vector = pd.DataFrame(
            [[raw_features[f] for f in feature_order]],
            columns=feature_order,
        )

        # Predict probability of conversion
        proba = lead_model.predict_proba(feature_vector)[0]
        conversion_prob = float(proba[1])

        # Assign a human-readable label
        if conversion_prob >= 0.6:
            label = "🟢 High — prioritize this lead"
        elif conversion_prob >= 0.3:
            label = "🟡 Medium — worth following up"
        else:
            label = "🔴 Low — lower priority"

        return jsonify({
            "score": round(conversion_prob * 100, 1),
            "label": label,
        })

    except Exception as e:
        return jsonify({"error": f"Scoring failed: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5000)
