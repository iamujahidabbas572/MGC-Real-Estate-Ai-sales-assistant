"""
Part 3 — Lead Scoring: Honest Baseline Model
=============================================
Trains a Random Forest classifier to predict which CRM leads will convert.

Data decisions (documented here and in README):

  DROPPED columns:
    - lead_id             : identifier, not a feature
    - created_at          : timestamp could be useful (day-of-week, recency)
                            but skipped for a quick baseline
    - crm_record_hash     : internal dedup hash, no predictive value
    - token_amount_received_pkr : THIS IS THE KEY DROP — it leaks the target.
                            A lead that received a token amount has almost
                            certainly already converted. Using it would give
                            inflated accuracy that would not generalize to
                            scoring new, unconverted leads.

  CLEANED:
    - city                : inconsistent casing (ISB/ISLAMABAD/Islamabad,
                            Rwp/RAWALPINDI/Rawalpindi, khi/KARACHI/Karachi,
                            etc.). Normalized to title case with known
                            abbreviation mappings.
    - bedrooms            : 39% missing. Imputed with median (2.0).
    - agent_experience_years, first_response_minutes, budget_pkr_lac :
                            missing values filled with column median.
    - area                : 5% missing. Filled with "Unknown".
    - Duplicate leads     : removed (kept first occurrence by crm_record_hash).

  KEPT features:
    source, city, area, property_type, budget_pkr_lac, bedrooms,
    first_response_minutes, calls_made, total_call_seconds,
    whatsapp_replies, site_visits, agent_experience_years,
    is_overseas, referred_by_existing_client, has_financing_approved

  METRIC:
    F1 Score (positive class) — chosen because of severe class imbalance:
    only ~6.9% of leads converted (634 out of 9,160). Plain accuracy would
    be 93% just by predicting "not converted" for everyone, which is useless
    for a sales team that needs to know WHO to call first.

Usage:
  python part3_lead_scoring.py
"""

import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "leads.csv"
MODEL_PATH = BASE_DIR / "lead_scoring_model.pkl"


# ---------------------------------------------------------------------------
# City normalization map — handles the messy CRM spellings
# ---------------------------------------------------------------------------
CITY_NORMALIZATION = {
    "islamabad": "Islamabad",
    "isb": "Islamabad",
    "rawalpindi": "Rawalpindi",
    "rwp": "Rawalpindi",
    "lahore": "Lahore",
    "karachi": "Karachi",
    "khi": "Karachi",
    "peshawar": "Peshawar",
    "faisalabad": "Faisalabad",
    "multan": "Multan",
    "gujranwala": "Gujranwala",
    "abbottabad": "Abbottabad",
}


def normalize_city(city_value):
    """Map messy city names to a canonical form."""
    if pd.isna(city_value):
        return "Unknown"
    cleaned = str(city_value).strip().lower()
    return CITY_NORMALIZATION.get(cleaned, city_value.strip().title())


def load_and_clean_data():
    """Load leads.csv, apply all cleaning decisions, and return X, y."""
    df = pd.read_csv(CSV_PATH)

    print(f"Raw dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Class balance: {df['converted'].value_counts().to_dict()}")
    print(f"Positive rate: {df['converted'].mean():.1%}\n")

    # ---- Remove duplicate leads (same crm_record_hash) ----
    before = len(df)
    df = df.drop_duplicates(subset=["crm_record_hash"], keep="first")
    print(f"Removed {before - len(df)} duplicate leads (by crm_record_hash)")

    # ---- Drop columns with no predictive value or target leakage ----
    drop_cols = [
        "lead_id",           # identifier
        "created_at",        # skipping for baseline
        "crm_record_hash",   # internal hash
        "token_amount_received_pkr",  # TARGET LEAKAGE — known post-conversion
    ]
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")

    # ---- Normalize city names ----
    df["city"] = df["city"].apply(normalize_city)
    print(f"Normalized city names ({df['city'].nunique()} unique cities)")

    # ---- Fill missing values ----
    df["area"] = df["area"].fillna("Unknown")
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["budget_pkr_lac"] = df["budget_pkr_lac"].fillna(df["budget_pkr_lac"].median())
    df["first_response_minutes"] = df["first_response_minutes"].fillna(
        df["first_response_minutes"].median()
    )
    df["agent_experience_years"] = df["agent_experience_years"].fillna(
        df["agent_experience_years"].median()
    )
    print(f"Imputed missing values (bedrooms, budget, response time, agent exp)")

    # ---- Encode categorical features ----
    categorical_cols = ["source", "city", "area", "property_type"]
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # ---- Split features and target ----
    y = df["converted"]
    X = df.drop(columns=["converted"])

    print(f"\nFinal dataset: {len(X)} rows, {len(X.columns)} features")
    print(f"Features: {list(X.columns)}\n")

    return X, y, label_encoders


def train_baseline_model(X, y):
    """Train a Random Forest classifier and report F1 score."""

    # Stratified split preserves the ~7% positive rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    print(f"Train positive rate: {y_train.mean():.1%} | Test positive rate: {y_test.mean():.1%}\n")

    # Random Forest with class_weight="balanced" to handle imbalance
    # This up-weights the minority class during training so the model
    # doesn't just learn to predict "not converted" for everything
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---- Report results ----
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("=" * 60)
    print(f"  PRIMARY METRIC: F1 Score (converted class) = {f1:.4f}")
    print("=" * 60)
    print(
        "\n  Why F1? The data is severely imbalanced (~7% positive).\n"
        "  Accuracy would be 93% by predicting 'not converted' for\n"
        "  everyone — useless for a sales team. F1 balances precision\n"
        "  (don't waste calls on bad leads) with recall (don't miss\n"
        "  real buyers).\n"
    )

    print("Full classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':>20} Predicted:No  Predicted:Yes")
    print(f"  {'Actual: No':>20}    {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"  {'Actual: Yes':>20}    {cm[1][0]:>6}        {cm[1][1]:>6}")

    # ---- Feature importance ----
    importances = model.feature_importances_
    feature_names = X_train.columns
    sorted_indices = np.argsort(importances)[::-1]

    print("\n\nTop 10 feature importances:")
    for rank, idx in enumerate(sorted_indices[:10], 1):
        print(f"  {rank:>2}. {feature_names[idx]:<30} {importances[idx]:.4f}")

    return model


def save_model(model, label_encoders):
    """Save the trained model and encoders for use in the web app."""
    artifact = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_order": [
            "source", "city", "area", "property_type", "budget_pkr_lac",
            "bedrooms", "first_response_minutes", "calls_made",
            "total_call_seconds", "whatsapp_replies", "site_visits",
            "agent_experience_years", "is_overseas",
            "referred_by_existing_client", "has_financing_approved",
        ],
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel saved to {MODEL_PATH}")


def main():
    print("=" * 60)
    print("  Part 3 — Lead Scoring Baseline Model")
    print("=" * 60)
    print()

    X, y, label_encoders = load_and_clean_data()
    model = train_baseline_model(X, y)
    save_model(model, label_encoders)

    print("\nDone. The model is ready for use in the web app (Part 4).")


if __name__ == "__main__":
    main()
