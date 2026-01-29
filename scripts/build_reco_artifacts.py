# final_projet/scripts/build_reco_artifacts.py
# Build TF-IDF artifacts for Streamlit usage (offline step).
#
# Run from project root:
#   python scripts/build_reco_artifacts.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PROCESSED = Path("data/data_processed")
FEATURES_PATH = DATA_PROCESSED / "movies_features.csv.gz"
OUT_DIR = DATA_PROCESSED / "reco"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = OUT_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = OUT_DIR / "tfidf_matrix.joblib"
INDEX_PATH = OUT_DIR / "tconst_index.csv"

# TF-IDF settings (good default for your "soup")
TFIDF_MAX_FEATURES = 120_000  # controls size; you can reduce if needed
TFIDF_NGRAM_RANGE = (1, 2)    # unigrams + bigrams helps for names like "hugh jackman"
TFIDF_MIN_DF = 2              # ignore terms that appear only once (noise)
TFIDF_MAX_DF = 0.90           # ignore extremely common tokens


def main() -> None:
    # Load features
    df = pd.read_csv(FEATURES_PATH)
    df["soup"] = df["soup"].fillna("").astype(str)

    # Keep only valid rows (should already be OK)
    df = df[df["tconst"].notna() & (df["tconst"].astype(str).str.strip() != "")]
    df = df[df["soup"].astype(str).str.strip() != ""].copy()
    df["tconst"] = df["tconst"].astype(str)

    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        stop_words=None,  # soup is controlled; do not remove tokens
    )

    X = vectorizer.fit_transform(df["soup"])

    # Save artifacts (joblib compress keeps size low)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    joblib.dump(X, MATRIX_PATH, compress=3)

    # Save index mapping row -> tconst
    df[["tconst"]].reset_index(drop=True).to_csv(INDEX_PATH, index=False)

    print(f"[OK] Vectorizer saved: {VECTORIZER_PATH}")
    print(f"[OK] Matrix saved:     {MATRIX_PATH} (shape={X.shape})")
    print(f"[OK] Index saved:      {INDEX_PATH} (rows={len(df):,})")


if __name__ == "__main__":
    main()
