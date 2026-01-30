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


BASE_DIR = Path(__file__).resolve().parents[1]  # dossier racine final_projet/
RECO_DIR = BASE_DIR / "data" / "data_processed" / "reco"


@dataclass(frozen=True)
class RecoArtifacts:
    vectorizer: object
    matrix: object  # matrice sparse (scipy) contenant les vecteurs TF-IDF
    tconst_list: List[str]
    tconst_to_row: dict


@st.cache_resource(show_spinner=False)
def load_reco_artifacts() -> RecoArtifacts:
    """
    Charge les artefacts nécessaires à la recommandation :
    - le vectorizer TF-IDF entraîné
    - la matrice TF-IDF du catalogue
    - l’index des tconst (ordre des lignes de la matrice)
    - un mapping tconst -> index de ligne
    Le tout est mis en cache par Streamlit pour éviter les rechargements.
    """
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
    Recommande des films à partir d’un film déjà présent dans le catalogue.
    Retourne une liste de tuples (tconst, score) correspondant aux films les plus similaires.
    """
    art = load_reco_artifacts()
    if query_tconst not in art.tconst_to_row:
        return []

    # Index et vecteur TF-IDF du film demandé
    q_idx = art.tconst_to_row[query_tconst]
    q_vec = art.matrix[q_idx]

    # Similarité cosinus entre le film cible et tous les films du catalogue
    sims = cosine_similarity(q_vec, art.matrix).ravel()

    # Exclusion du film lui-même (pour éviter qu’il sorte en top 1)
    sims[q_idx] = -1.0

    # Cas limite
    if top_n <= 0:
        return []

    # Récupération des top_n indices (plus rapides qu’un tri complet)
    top_idx = np.argpartition(-sims, range(min(top_n, len(sims))))[:top_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(art.tconst_list[i], float(sims[i])) for i in top_idx]


def recommend_by_soup(query_soup: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Recommande des films à partir d’un film externe (non présent dans le catalogue),
    en utilisant un texte descriptif 'soup' (genres + cast + réalisateur, etc.).
    Le texte est vectorisé avec le TF-IDF déjà entraîné (pas de refit),
    puis comparé à la matrice du catalogue via similarité cosinus.
    """
    art = load_reco_artifacts()
    query_soup = (query_soup or "").strip().lower()
    if not query_soup:
        return []

    # Vectorisation du texte d’entrée avec le vectorizer TF-IDF existant
    q_vec = art.vectorizer.transform([query_soup])

    # Similarité cosinus entre le film externe et tous les films du catalogue
    sims = cosine_similarity(q_vec, art.matrix).ravel()

    # Sélection des top_n films les plus similaires
    top_idx = np.argpartition(-sims, range(min(top_n, len(sims))))[:top_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(art.tconst_list[i], float(sims[i])) for i in top_idx]
