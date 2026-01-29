# final_projet/streamlit_app/recommender.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]  # final_projet/
RECO_DIR = BASE_DIR / "data" / "data_processed" / "reco"


@dataclass(frozen=True)
class RecoArtifacts:
    vectorizer: object
    matrix: object  # scipy sparse matrix
    tconst_list: List[str]
    tconst_to_row: dict


@st.cache_resource(show_spinner=False)
def load_reco_artifacts() -> RecoArtifacts:
    vectorizer = joblib.load(RECO_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(RECO_DIR / "tfidf_matrix.joblib")

    idx = pd.read_csv(RECO_DIR / "tconst_index.csv")
    tconst_list = idx["tconst"].astype(str).tolist()
    tconst_to_row = {t: i for i, t in enumerate(tconst_list)}

    return RecoArtifacts(
        vectorizer=vectorizer,
        matrix=matrix,
        tconst_list=tconst_list,
        tconst_to_row=tconst_to_row,
    )


def recommend_by_tconst(query_tconst: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Returns list of (tconst, score) recommended for a film already in the catalog.
    """
    art = load_reco_artifacts()
    if query_tconst not in art.tconst_to_row:
        return []

    q_idx = art.tconst_to_row[query_tconst]
    q_vec = art.matrix[q_idx]

    # cosine similarity between query and all items
    sims = cosine_similarity(q_vec, art.matrix).ravel()

    # exclude itself
    sims[q_idx] = -1.0

    # get top_n indices
    if top_n <= 0:
        return []

    top_idx = np.argpartition(-sims, range(min(top_n, len(sims))))[:top_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(art.tconst_list[i], float(sims[i])) for i in top_idx]


def recommend_by_soup(query_soup: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Returns list of (tconst, score) recommended for an external film (not in catalog),
    using its soup transformed with the fitted vectorizer (NO refit).
    """
    art = load_reco_artifacts()
    query_soup = (query_soup or "").strip().lower()
    if not query_soup:
        return []

    q_vec = art.vectorizer.transform([query_soup])
    sims = cosine_similarity(q_vec, art.matrix).ravel()

    top_idx = np.argpartition(-sims, range(min(top_n, len(sims))))[:top_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(art.tconst_list[i], float(sims[i])) for i in top_idx]
