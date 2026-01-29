# src/market_app.py
# ============================================================
# market_app.py
# -------------
# Module Streamlit: Étude de marché (INSEE + CNC)
#
# Objectif:
# - Reprendre ton Streamlit "étude de marché" et le mettre dans une fonction
#   render_market() pour qu'il soit appelé depuis app.py (hub).
#
# Changement principal vs ton ancien code:
# - On ne charge PLUS depuis GitHub raw.
# - On charge depuis les fichiers du repo (chemins relatifs) -> compatible Streamlit.io.
#
# IMPORTANT:
# - Ne mélange pas DB/ML ici (app.py a déjà les menus dédiés).
# - On garde les graphes tels quels (logique identique).
# ============================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # projet_2_final/
DATA_BASE = BASE_DIR / "data"


# ============================================================
# Chargement (cache)
# ============================================================

@st.cache_data(show_spinner="Chargement des données INSEE + CNC...")
def load_market_data() -> dict:
    # -------- INSEE --------
    df_age = pd.read_csv(DATA_BASE / "INSEE" / "insee_age.csv")
    df_menages = pd.read_csv(DATA_BASE / "INSEE" / "insee_menages.csv")
    df_pauvrete = pd.read_csv(DATA_BASE / "INSEE" / "insee_pauvrete.csv")
    df_sal_age = pd.read_csv(DATA_BASE / "INSEE" / "insee_salaires_age.csv")
    df_sal_csp = pd.read_csv(DATA_BASE / "INSEE" / "insee_salaires_csp.csv")

    # -------- CNC --------
    df_cnc_ecrans = pd.read_csv(DATA_BASE / "CNC" / "cnc_ecrans.csv")
    df_cnc_entrees = pd.read_csv(DATA_BASE / "CNC" / "cnc_entrees.csv")
    df_cnc_etab = pd.read_csv(DATA_BASE / "CNC" / "cnc_etablissements.csv")
    df_cnc_fauteuils = pd.read_csv(DATA_BASE / "CNC" / "cnc_fauteuils.csv")
    df_cnc_indice = pd.read_csv(DATA_BASE / "CNC" / "cnc_indice_frequentation.csv")
    df_cnc_seances = pd.read_csv(DATA_BASE / "CNC" / "cnc_seances.csv")
    df_cnc_taux_occ = pd.read_csv(DATA_BASE / "CNC" / "cnc_taux_occupation.csv")

    return {
        "df_age": df_age,
        "df_menages": df_menages,
        "df_pauvrete": df_pauvrete,
        "df_sal_age": df_sal_age,
        "df_sal_csp": df_sal_csp,
        "df_cnc_ecrans": df_cnc_ecrans,
        "df_cnc_entrees": df_cnc_entrees,
        "df_cnc_etab": df_cnc_etab,
        "df_cnc_fauteuils": df_cnc_fauteuils,
        "df_cnc_indice": df_cnc_indice,
        "df_cnc_seances": df_cnc_seances,
        "df_cnc_taux_occ": df_cnc_taux_occ,
    }


# ============================================================
# Helpers (Nettoyage)
# ============================================================

def clean_spaces(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\u202f", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("nan", np.nan)
    )


# ============================================================
# Graph 1
# ============================================================

def graph_1(df_age: pd.DataFrame):
    df_age_clean = df_age.copy()
    df_age_clean = df_age_clean.iloc[1:].reset_index(drop=True)
    df_age_clean.columns = ["age", "pop_2011", "pct_2011", "pop_2016", "pct_2016", "pop_2022", "pct_2022"]

    for col in ["pop_2011", "pop_2016", "pop_2022"]:
        df_age_clean[col] = pd.to_numeric(clean_spaces(df_age_clean[col]), errors="coerce")

    for col in ["pct_2011", "pct_2016", "pct_2022"]:
        df_age_clean[col] = (
            df_age_clean[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace("nan", np.nan)
        )
        df_age_clean[col] = pd.to_numeric(df_age_clean[col], errors="coerce")

    df_age_2022 = df_age_clean[df_age_clean["age"].str.lower() != "ensemble"].copy()
    ordre = ["0 à 14 ans", "15 à 29 ans", "30 à 44 ans", "45 à 59 ans", "60 à 74 ans", "75 ans ou plus"]
    df_age_2022["age"] = pd.Categorical(df_age_2022["age"], categories=ordre, ordered=True)
    df_age_2022 = df_age_2022.sort_values("age")

    st.subheader("Graph 1 — Répartition de la population par âge (2022)")
    st.dataframe(df_age_2022[["age", "pop_2022", "pct_2022"]], use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_age_2022["age"].astype(str), df_age_2022["pct_2022"])
    ax.set_title("Creuse — Répartition de la population par âge (2022)")
    ax.set_xlabel("Tranches d'âge")
    ax.set_ylabel("Part de la population (%)")
    plt.xticks(rotation=30, ha="right")

    for i, v in enumerate(df_age_2022["pct_2022"].values):
        if pd.notna(v):
            ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    return df_age_clean


# ============================================================
# Graph 2
# ============================================================

def graph_2(df_age_clean: pd.DataFrame):
    df = df_age_clean.copy()
    df = df[df["age"].str.lower() != "ensemble"]

    def groupe_age(age):
        if age in ["0 à 14 ans", "15 à 29 ans"]:
            return "-20 ans"
        elif age in ["30 à 44 ans", "45 à 59 ans"]:
            return "20–59 ans"
        elif age in ["60 à 74 ans", "75 ans ou plus"]:
            return "60 ans et +"
        return None

    df["groupe"] = df["age"].apply(groupe_age)
    df = df.dropna(subset=["groupe"])

    evolution = (
        df.groupby("groupe")[["pct_2011", "pct_2016", "pct_2022"]]
        .sum()
        .reset_index()
    )

    evolution_long = evolution.melt(
        id_vars="groupe",
        var_name="annee",
        value_name="pourcentage"
    )
    evolution_long["annee"] = evolution_long["annee"].str.replace("pct_", "").astype(int)

    st.subheader("Graph 2 — Évolution des grandes tranches d’âge (2011–2022)")

    fig, ax = plt.subplots(figsize=(10, 5))
    for groupe in evolution_long["groupe"].unique():
        subset = evolution_long[evolution_long["groupe"] == groupe]
        ax.plot(subset["annee"], subset["pourcentage"], marker="o", label=groupe)

    ax.set_title("Creuse — Évolution des grandes tranches d’âge (2011–2022)")
    ax.set_xlabel("Année")
    ax.set_ylabel("Part de la population (%)")
    ax.set_xticks([2011, 2016, 2022])
    ax.legend(title="Tranche d'âge")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


# ============================================================
# Graph 3
# ============================================================

def graph_3(df_menages: pd.DataFrame):
    st.subheader("Graph 3 — Composition des ménages en Creuse (2022)")

    df = df_menages.copy()
    df.columns = [
        "type_menage",
        "nb_2011", "pct_2011",
        "nb_2016", "pct_2016",
        "nb_2022", "pct_2022",
        "pop_2011", "pop_2016", "pop_2022"
    ]

    df_2022 = df[
        df["type_menage"].isin([
            "Ménages d'une personne",
            "Autres ménages sans famille",
            "Un couple sans enfant",
            "Un couple avec enfant(s)",
            "Une famille monoparentale"
        ])
    ].copy()

    df_2022["pct_2022"] = (
        df_2022["pct_2022"].astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(df_2022[["type_menage", "pct_2022"]], use_container_width=True)

    with col2:
        labels = df_2022["type_menage"]
        sizes = df_2022["pct_2022"]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.8
        )
        centre_circle = plt.Circle((0, 0), 0.55, fc="white")
        ax.add_artist(centre_circle)
        ax.set_title("Composition des ménages en Creuse (2022)")
        plt.tight_layout()
        st.pyplot(fig)


# ============================================================
# Graph 4
# ============================================================

def graph_4(df_pauvrete: pd.DataFrame):
    st.subheader("Graph 4 — Taux de pauvreté par tranche d’âge (Creuse, 2021)")

    df = df_pauvrete.copy()
    df.columns = ["tranche_age", "taux_pauvrete"]
    df = df[df["taux_pauvrete"] != "Taux en %"]

    df["taux_pauvrete"] = (
        df["taux_pauvrete"].astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df_plot = df[df["tranche_age"] != "Ensemble"].copy()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df_plot, use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df_plot["taux_pauvrete"], df_plot["tranche_age"], s=120)

        for x, y in zip(df_plot["taux_pauvrete"], df_plot["tranche_age"]):
            ax.text(x + 0.3, y, f"{x:.1f}%", va="center", fontsize=9)

        ax.set_title("Taux de pauvreté par tranche d’âge en Creuse (2021)")
        ax.set_xlabel("Taux de pauvreté (%)")
        ax.set_ylabel("Tranche d’âge")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)


# ============================================================
# Graph 5
# ============================================================

def graph_5(df_sal_age: pd.DataFrame):
    st.subheader("Graph 5 — Salaire net mensuel moyen par CSP (2023)")

    df_sal = df_sal_age.copy()
    df_sal.columns = ["categorie", "ensemble", "femmes", "hommes"]

    for col in ["ensemble", "femmes", "hommes"]:
        df_sal[col] = pd.to_numeric(clean_spaces(df_sal[col]), errors="coerce")

    df_sal_clean = df_sal[df_sal["categorie"].str.lower() != "ensemble"].copy()
    ordre_cat = ["Cadres*", "Professions intermédiaires", "Employés"]
    df_sal_clean["categorie"] = pd.Categorical(df_sal_clean["categorie"], categories=ordre_cat, ordered=True)
    df_sal_clean = df_sal_clean.sort_values("categorie")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df_sal_clean[["categorie", "ensemble"]], use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(df_sal_clean["categorie"].astype(str), df_sal_clean["ensemble"])
        ax.set_title("Salaire net mensuel moyen par CSP (2023)")
        ax.set_xlabel("Catégories socioprofessionnelles")
        ax.set_ylabel("Salaire net moyen (€)")
        plt.xticks(rotation=20, ha="right")

        ymax = np.nanmax(df_sal_clean["ensemble"].values)
        ax.set_ylim(0, ymax * 1.12)

        ax.bar_label(
            bars,
            labels=[f"{v:.0f} €" if np.isfinite(v) else "" for v in df_sal_clean["ensemble"].values],
            padding=3,
            fontsize=9
        )

        plt.tight_layout()
        st.pyplot(fig)


# ============================================================
# Graph 6
# ============================================================

def graph_6(df_cnc_ecrans: pd.DataFrame):
    st.subheader("Graph 6 — Nombre d’écrans de cinéma en Creuse (1966–2024)")

    df_ecr = df_cnc_ecrans.copy()
    df_23 = df_ecr[df_ecr["dep_code"] == "23"].copy()

    df_23_years = df_23.drop(columns=["dep_code", "dep_nom"])
    df_23_t = df_23_years.T.reset_index()
    df_23_t.columns = ["annee", "nb_ecrans"]

    df_23_t["annee"] = pd.to_numeric(df_23_t["annee"], errors="coerce")
    df_23_t["nb_ecrans"] = pd.to_numeric(df_23_t["nb_ecrans"], errors="coerce")
    df_23_t = df_23_t.sort_values("annee")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df_23_t.tail(15), use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_23_t["annee"], df_23_t["nb_ecrans"], marker="o")
        ax.set_title("Creuse — Nombre d’écrans de cinéma (1966–2024)")
        ax.set_xlabel("Année")
        ax.set_ylabel("Nombre d’écrans")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)


# ============================================================
# Graph 7 + 8
# ============================================================

def graph_7_8(df_cnc_entrees: pd.DataFrame, df_cnc_indice: pd.DataFrame):
    st.subheader("Graph 7 + 8 — Fréquentation (volume absolu + position relative)")

    # ---- Graph 7 : Entrées Creuse ----
    df_ent = df_cnc_entrees.copy()
    df_ent["dep_code"] = df_ent["dep_code"].astype(str).str.strip()
    creuse_ent = df_ent[df_ent["dep_code"] == "23"]

    year_cols_ent = [c for c in creuse_ent.columns if c.isdigit()]

    ent_long = creuse_ent.melt(
        id_vars=["dep_code", "dep_nom"],
        value_vars=year_cols_ent,
        var_name="annee",
        value_name="entrees"
    )

    ent_long["annee"] = ent_long["annee"].astype(int)
    ent_long["entrees"] = (
        ent_long["entrees"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    ent_long["entrees"] = pd.to_numeric(ent_long["entrees"], errors="coerce")
    ent_long = ent_long.dropna().sort_values("annee")

    # ---- Graph 8 : Indice Creuse vs TOTAL ----
    df_ind = df_cnc_indice.copy()
    df_ind["dep_nom"] = df_ind["dep_nom"].astype(str).str.strip()

    ind_sel = df_ind[df_ind["dep_nom"].isin(["CREUSE", "TOTAL"])]
    year_cols_ind = [c for c in ind_sel.columns if c.isdigit()]

    ind_long = ind_sel.melt(
        id_vars=["dep_nom"],
        value_vars=year_cols_ind,
        var_name="annee",
        value_name="indice"
    )

    ind_long["annee"] = ind_long["annee"].astype(int)
    ind_long["indice"] = (
        ind_long["indice"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    ind_long["indice"] = pd.to_numeric(ind_long["indice"], errors="coerce")
    ind_long = ind_long.dropna().sort_values(["dep_nom", "annee"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Haut : Entrées
    axes[0].plot(ent_long["annee"], ent_long["entrees"], marker="o", linewidth=2)
    axes[0].axvline(2020, linestyle="--", alpha=0.6)
    axes[0].set_title("Entrées en salles de cinéma en Creuse (historique)")
    axes[0].set_ylabel("Nombre d’entrées")
    axes[0].grid(alpha=0.3)

    # Bas : Indice
    for name, d in ind_long.groupby("dep_nom"):
        axes[1].plot(d["annee"], d["indice"], marker="o", label=name)

    axes[1].axvline(2020, linestyle="--", alpha=0.6)
    axes[1].set_title("Indice de fréquentation – Creuse vs moyenne nationale")
    axes[1].set_xlabel("Année")
    axes[1].set_ylabel("Indice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(
        "Fréquentation des salles de cinéma en Creuse\n"
        "Volume absolu et position relative",
        fontsize=14
    )

    plt.tight_layout()
    st.pyplot(fig)


# ============================================================
# Graph 9 (Personas)
# ============================================================

def graph_9():
    st.subheader("Graph 9 — Synthèse Personas (tableau)")

    personas = pd.DataFrame({
        "Persona": [
            "Seniors creusois (60+)",
            "Adultes actifs (30–59)",
            "Familles (parents + enfants)",
            "Jeunes (15–29)"
        ],
        "Poids (qualitatif)": ["Fort", "Moyen", "Moyen", "Faible"],
        "Genres recommandés": [
            "Comédie française, drame, patrimoine, documentaires",
            "Thriller, comédie grand public, drame, films événements",
            "Animation, comédie familiale, aventure",
            "Films événements, cinéma de genre, VO/séances spéciales"
        ],
        "Fréquence cible": [
            "1–2 séances/mois",
            "1 séance/trimestre à 1 séance/mois (selon offre)",
            "1 séance/mois (week-end/vacances)",
            "Ponctuelle (événements / séances spéciales)"
        ]
    })

    st.dataframe(personas, use_container_width=True)


# ============================================================
# Entrée module
# ============================================================

def render_market():
    """
    Page "Étude de marché" appelée depuis app.py.
    """
    data = load_market_data()

    df_age = data["df_age"]
    df_menages = data["df_menages"]
    df_pauvrete = data["df_pauvrete"]
    df_sal_age = data["df_sal_age"]
    df_cnc_ecrans = data["df_cnc_ecrans"]
    df_cnc_entrees = data["df_cnc_entrees"]
    df_cnc_indice = data["df_cnc_indice"]

    st.sidebar.markdown("### Étude de marché — Graphiques")

    market_graph = st.sidebar.radio(
        "Sélection",
        [
            "Graph 1 — Âges (2022)",
            "Graph 2 — Vieillissement (2011–2022)",
            "Graph 3 — Ménages (2022)",
            "Graph 4 — Pauvreté (2021)",
            "Graph 5 — Salaires (2023)",
            "Graph 6 — Écrans (1966–2024)",
            "Graph 7+8 — Entrées + Indice",
            "Graph 9 — Personas",
            "Tout afficher",
        ],
        index=0,
        key="market_graph_selector",
    )

    if market_graph == "Tout afficher":
        _df_age_clean = graph_1(df_age)
        graph_2(_df_age_clean)
        graph_3(df_menages)
        graph_4(df_pauvrete)
        graph_5(df_sal_age)
        graph_6(df_cnc_ecrans)
        graph_7_8(df_cnc_entrees, df_cnc_indice)
        graph_9()
        return

    if market_graph == "Graph 1 — Âges (2022)":
        graph_1(df_age)

    elif market_graph == "Graph 2 — Vieillissement (2011–2022)":
        _df_age_clean = graph_1(df_age)
        graph_2(_df_age_clean)

    elif market_graph == "Graph 3 — Ménages (2022)":
        graph_3(df_menages)

    elif market_graph == "Graph 4 — Pauvreté (2021)":
        graph_4(df_pauvrete)

    elif market_graph == "Graph 5 — Salaires (2023)":
        graph_5(df_sal_age)

    elif market_graph == "Graph 6 — Écrans (1966–2024)":
        graph_6(df_cnc_ecrans)

    elif market_graph == "Graph 7+8 — Entrées + Indice":
        graph_7_8(df_cnc_entrees, df_cnc_indice)

    elif market_graph == "Graph 9 — Personas":
        graph_9()
        
page_market = render_market
