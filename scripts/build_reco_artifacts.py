# final_projet/scripts/build_reco_artifacts.py
# Génère les artefacts TF-IDF pour l’application Streamlit (étape hors ligne).

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

# Paramètres TF-IDF (bons réglages par défaut pour une colonne "soup")
TFIDF_MAX_FEATURES = 120_000  # contrôle la taille du vocabulaire (réduire si besoin)
TFIDF_NGRAM_RANGE = (1, 2)    # unigrams + bigrams (utile pour des noms/prénoms)
TFIDF_MIN_DF = 2              # ignore les termes vus une seule fois (bruit)
TFIDF_MAX_DF = 0.90           # ignore les tokens trop fréquents


def main() -> None:
    """
    Charge le dataset de features (tconst + soup), entraîne un TF-IDF sur la colonne soup,
    puis sauvegarde :
    - le vectorizer entraîné
    - la matrice TF-IDF du catalogue
    - l’index tconst dans le même ordre que les lignes de la matrice
    """
    # Chargement des features
    df = pd.read_csv(FEATURES_PATH)
    df["soup"] = df["soup"].fillna("").astype(str)

    # On garde uniquement les lignes valides :
    # - tconst non vide
    # - soup non vide
    df = df[df["tconst"].notna() & (df["tconst"].astype(str).str.strip() != "")]
    df = df[df["soup"].astype(str).str.strip() != ""].copy()
    df["tconst"] = df["tconst"].astype(str)

    # Construction TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        stop_words=None,  # la soup est déjà contrôlée ; on ne retire pas de tokens
    )

    # Entraînement + transformation : création de la matrice sparse TF-IDF
    X = vectorizer.fit_transform(df["soup"])

    # Sauvegarde des artefacts (joblib + compression pour limiter la taille)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    joblib.dump(X, MATRIX_PATH, compress=3)

    # Sauvegarde de l’index : correspondance ligne -> tconst
    df[["tconst"]].reset_index(drop=True).to_csv(INDEX_PATH, index=False)

    print(f"[OK] Vectorizer saved: {VECTORIZER_PATH}")
    print(f"[OK] Matrix saved:     {MATRIX_PATH} (shape={X.shape})")
    print(f"[OK] Index saved:      {INDEX_PATH} (rows={len(df):,})")


if __name__ == "__main__":
    main()
