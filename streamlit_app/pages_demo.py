# streamlit_app/pages_demo.py
# UI "Netflix-like" – Home / Recherche / Fiche film
# Lancer depuis la racine : streamlit run streamlit_app/app.py
#
# Inclus :
# - Navigation (boutons) sur chaque page
# - Titres FR via TMDB si dispo (fallback IMDb primaryTitle)
# - Posters (pas de via.placeholder.com, placeholder HTML local)
# - Recherche : fallback TMDB si pas de match local (film + acteur/actrice)
# - Fiche : synopsis FR + recos en vignettes + rerank now/upcoming
# - Casting : déduplication affichage

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from recommender import recommend_by_soup, recommend_by_tconst
from tmdb_client import (
    build_now_upcoming_imdb_id_sets,
    build_soup_from_tmdb_bundle,
    get_now_playing_fr,
    get_or_fetch_bundle_by_imdb_id,
    get_tmdb_details_bundle,
    get_upcoming_fr,
    tmdb_to_display_record,
    search_movie_fr,
    search_person_fr,
    get_person_movie_credits,
)

POSTER_BASE = "https://image.tmdb.org/t/p/w342"
BACKDROP_BASE = "https://image.tmdb.org/t/p/w780"


# =============================
# Data
# =============================
@st.cache_data
def load_movies_local() -> pd.DataFrame:
    return pd.read_csv("data/data_processed/movies_local.csv.gz")


@st.cache_data(show_spinner=False)
def load_now_upcoming_lists() -> Tuple[list, list]:
    now_items = get_now_playing_fr(pages=1, use_cache=True)
    up_items = get_upcoming_fr(pages=1, use_cache=True)
    return now_items, up_items


@st.cache_data(show_spinner=False)
def load_now_upcoming_sets() -> Tuple[set, set]:
    now_ids, up_ids = build_now_upcoming_imdb_id_sets(pages_now=1, pages_upcoming=1)
    return now_ids, up_ids


# =============================
# State
# =============================
def _init_state() -> None:
    st.session_state.setdefault("demo_page", "home")  # home | search | film
    st.session_state.setdefault("selected_tmdb_id", None)
    st.session_state.setdefault("selected_tconst", None)
    st.session_state.setdefault("film_query", "")
    st.session_state.setdefault("actor_query", "")


def _go(page: str) -> None:
    st.session_state["demo_page"] = page


def _open_tmdb(tmdb_id: int) -> None:
    st.session_state["selected_tmdb_id"] = int(tmdb_id)
    st.session_state["selected_tconst"] = None
    _go("film")


def _open_local(tconst: str) -> None:
    st.session_state["selected_tconst"] = str(tconst)
    st.session_state["selected_tmdb_id"] = None
    _go("film")


def _on_film_change() -> None:
    st.session_state["actor_query"] = ""


def _on_actor_change() -> None:
    st.session_state["film_query"] = ""


# =============================
# UI helpers
# =============================
def _nav_bar() -> None:
    nav = st.columns([1, 1, 1, 5])
    nav[0].button("🏠 Accueil", on_click=_go, args=("home",), use_container_width=True)
    nav[1].button("🔎 Recherche", on_click=_go, args=("search",), use_container_width=True)
    nav[2].button("🎬 Film", on_click=_go, args=("film",), use_container_width=True)


def _poster(path: Optional[str]) -> Optional[str]:
    return f"{POSTER_BASE}{path}" if path else None


def _backdrop(path: Optional[str]) -> Optional[str]:
    return f"{BACKDROP_BASE}{path}" if path else None


def _short(txt: str, n: int = 80) -> str:
    t = (txt or "").strip()
    if not t:
        return ""
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _dedup_cast(cast: str) -> str:
    seen, out = set(), []
    for c in (cast or "").split("|"):
        c = c.strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return " | ".join(out)


def _placeholder() -> None:
    st.markdown(
        """
        <div style="aspect-ratio:2/3;
        background:rgba(255,255,255,0.06);
        border:1px solid rgba(255,255,255,0.12);
        display:flex;align-items:center;justify-content:center;
        color:rgba(255,255,255,0.55);font-size:12px;border-radius:8px;">
        No poster
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_cards(cards: List[Dict[str, Any]], key_prefix: str, cols: int = 5) -> None:
    if not cards:
        st.info("Aucun élément à afficher.")
        return

    grid = st.columns(cols)
    for i, c in enumerate(cards):
        with grid[i % cols]:
            if c.get("poster"):
                st.image(c["poster"], use_container_width=True)
            else:
                _placeholder()

            badge = c.get("badge")
            if badge:
                st.caption(f"**{badge}** — {c.get('subtitle','')}")
            else:
                st.caption(c.get("subtitle", ""))

            st.write(f"**{c.get('title','')}**")

            if c.get("kind") == "tmdb":
                st.button(
                    "Ouvrir",
                    key=f"{key_prefix}_tmdb_{c['id']}_{i}",
                    on_click=_open_tmdb,
                    args=(int(c["id"]),),
                    use_container_width=True,
                )
            else:
                st.button(
                    "Ouvrir",
                    key=f"{key_prefix}_local_{c['id']}_{i}",
                    on_click=_open_local,
                    args=(str(c["id"]),),
                    use_container_width=True,
                )


def _local_to_cards(df: pd.DataFrame, badge: Optional[str] = "Local") -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        main_genre = str(r.genres).split(",")[0] if pd.notna(r.genres) else ""
        cards.append(
            {
                "kind": "local",
                "id": str(r.tconst),
                "title": str(r.primaryTitle),
                "subtitle": f"{int(r.startYear)} • {main_genre} • IMDb {r.averageRating}",
                "poster": None,
                "badge": badge,
            }
        )
    return cards


def _tmdb_to_cards(items, badge: str) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for it in items:
        year = (getattr(it, "release_date", "") or "")[:4]
        cards.append(
            {
                "kind": "tmdb",
                "id": int(getattr(it, "tmdb_id")),
                "title": getattr(it, "title", ""),
                "subtitle": f"{year} • {_short(getattr(it, 'overview', '') or '', 70)}",
                "poster": _poster(getattr(it, "poster_path", None)),
                "badge": badge,
            }
        )
    return cards


def _enrich_local(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrichit les cartes locales via TMDB :
    - titre FR si dispo
    - poster
    - (optionnel) synopsis FR court
    """
    out: List[Dict[str, Any]] = []
    for c in cards:
        tconst = str(c["id"])
        bundle = get_or_fetch_bundle_by_imdb_id(tconst, use_cache=True)
        if bundle:
            rec = tmdb_to_display_record(bundle)

            title_fr = (rec.get("primaryTitle") or "").strip()
            if title_fr:
                c["title"] = title_fr

            c["poster"] = _poster(rec.get("poster_path"))

            ov = (rec.get("overview_fr") or "").strip()
            if ov:
                c["subtitle"] = f"{c['subtitle']} • {_short(ov, 70)}"

        out.append(c)
    return out


def _rerank(tconsts: List[str], now_ids: set, up_ids: set) -> List[str]:
    return (
        [t for t in tconsts if t in now_ids]
        + [t for t in tconsts if t in up_ids and t not in now_ids]
        + [t for t in tconsts if t not in now_ids and t not in up_ids]
    )


# =============================
# Pages
# =============================
def _page_home(df: pd.DataFrame) -> None:
    _nav_bar()
    st.title("Site démo")
    st.divider()

    with st.spinner("Chargement TMDB (FR)…"):
        now_items, up_items = load_now_upcoming_lists()

    st.subheader("À l'affiche")
    _render_cards(_tmdb_to_cards(now_items, badge="À l'affiche"), key_prefix="home_now")

    st.subheader("Bientôt")
    _render_cards(_tmdb_to_cards(up_items, badge="Bientôt"), key_prefix="home_up")

    st.divider()
    st.subheader("Top 10 par genre (catalogue local)")

    genres = sorted({g for gs in df["genres"] for g in str(gs).split(",") if g})
    selected_genre = st.selectbox("Choisir un genre", genres, key="home_genre")

    top = (
        df[df["genres"].str.contains(selected_genre, na=False)]
        .sort_values(["numVotes", "averageRating"], ascending=False)
        .head(10)
    )
    cards = _enrich_local(_local_to_cards(top, badge="Local"))
    _render_cards(cards, key_prefix="home_local")


def _page_search(df: pd.DataFrame) -> None:
    _nav_bar()
    st.title("Recherche")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Rechercher par film", key="film_query", on_change=_on_film_change, placeholder="ex: spiderman")
    with col2:
        st.text_input(
            "Rechercher par acteur/actrice",
            key="actor_query",
            on_change=_on_actor_change,
            placeholder="ex: tom hanks",
        )

    st.caption("Résultats : affichage des 5 plus récents (tri par année/date décroissante).")
    film_q = (st.session_state.get("film_query") or "").strip()
    actor_q = (st.session_state.get("actor_query") or "").strip()

    st.divider()

    if film_q:
        st.subheader("Résultats films (Top 5 récents)")

        res = df[df["primaryTitle"].str.contains(film_q, case=False, na=False)].copy()
        res = res.sort_values("startYear", ascending=False).head(5)

        if len(res) > 0:
            cards = _enrich_local(_local_to_cards(res, badge="Local"))
            _render_cards(cards, key_prefix="search_local_film")
        else:
            with st.spinner("Recherche TMDB…"):
                tmdb_hits = search_movie_fr(film_q, pages=1)

            tmdb_hits = sorted(tmdb_hits, key=lambda x: (getattr(x, "release_date", "") or ""), reverse=True)[:5]
            if tmdb_hits:
                _render_cards(_tmdb_to_cards(tmdb_hits, badge="TMDB"), key_prefix="search_tmdb_film")
            else:
                st.info("Aucun résultat (local ou TMDB).")

    elif actor_q:
        st.subheader("Résultats acteur/actrice (Top 5 récents)")

        res = df[df["cast_names_top5"].fillna("").str.contains(actor_q, case=False, na=False)].copy()
        res = res.sort_values("startYear", ascending=False).head(5)

        if len(res) > 0:
            cards = _enrich_local(_local_to_cards(res, badge="Local"))
            _render_cards(cards, key_prefix="search_local_actor")
        else:
            with st.spinner("Recherche TMDB (personne)…"):
                people = search_person_fr(actor_q, pages=1)

            if not people:
                st.info("Aucun résultat (local ou TMDB).")
                return

            person_id = int(people[0].get("id"))
            with st.spinner("Chargement films de la personne…"):
                credits = get_person_movie_credits(person_id)

            credits = sorted(credits, key=lambda x: (getattr(x, "release_date", "") or ""), reverse=True)[:5]
            if credits:
                _render_cards(_tmdb_to_cards(credits, badge="TMDB"), key_prefix="search_tmdb_actor")
            else:
                st.info("Aucun film trouvé pour cette personne (TMDB).")

    else:
        st.info("Tape un titre OU un acteur/actrice. L’autre champ se vide automatiquement.")


def _page_film(df: pd.DataFrame) -> None:
    _nav_bar()
    st.title("Film")
    st.divider()

    now_ids, up_ids = load_now_upcoming_sets()
    tmdb_id = st.session_state.get("selected_tmdb_id")
    tconst = st.session_state.get("selected_tconst")

    if tmdb_id is None and tconst is None:
        st.info("Sélectionne un film depuis Accueil/Recherche, ou choisis un film local ci-dessous.")
        pick = st.selectbox("Film local", df["primaryTitle"].sort_values().tolist())
        tconst_pick = df[df["primaryTitle"] == pick].iloc[0]["tconst"]
        st.button("Ouvrir la fiche", on_click=_open_local, args=(tconst_pick,), type="primary")
        return

    # --------- MODE TMDB ----------
    if tmdb_id is not None:
        bundle = get_tmdb_details_bundle(int(tmdb_id), use_cache=True)
        rec = tmdb_to_display_record(bundle)

        poster_url = _poster(rec.get("poster_path"))
        backdrop_url = _backdrop(rec.get("backdrop_path"))

        left, right = st.columns([1, 3])
        with left:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                _placeholder()

        with right:
            st.subheader(rec.get("primaryTitle", ""))
            st.caption(f"{rec.get('startYear','')} • {rec.get('genres','')} • {rec.get('runtimeMinutes','')} min")
            st.write(f"**Réalisateur :** {rec.get('director_names','')}")
            st.write(f"**Casting :** {_dedup_cast(rec.get('cast_names_top5',''))}")
            st.write(
                f"**Note TMDB :** {rec.get('tmdb_vote_average','')} / 10  (votes: {rec.get('tmdb_vote_count','')})"
            )

        if backdrop_url:
            st.image(backdrop_url, use_container_width=True)

        if rec.get("overview_fr"):
            st.subheader("Synopsis (FR)")
            st.write(rec["overview_fr"])

        st.divider()

        imdb_id = (rec.get("imdb_id") or "").strip()
        if imdb_id and imdb_id in set(df["tconst"].astype(str)):
            raw = recommend_by_tconst(imdb_id, top_n=30)
            tconsts = [t for (t, _) in raw]
        else:
            soup = build_soup_from_tmdb_bundle(bundle)
            raw = recommend_by_soup(soup, top_n=30)
            tconsts = [t for (t, _) in raw]

        ranked = _rerank(tconsts, now_ids, up_ids)[:5]
        rec_df = df[df["tconst"].isin(ranked)].copy()
        order = {t: i for i, t in enumerate(ranked)}
        rec_df["rank"] = rec_df["tconst"].map(order)
        rec_df = rec_df.sort_values("rank")

        st.subheader("Recommandations (priorité à l'affiche / bientôt)")
        cards = _enrich_local(_local_to_cards(rec_df, badge=None))
        _render_cards(cards, key_prefix="film_reco_tmdb")

        return

    # --------- MODE LOCAL ----------
    film = df[df["tconst"] == str(tconst)].iloc[0]
    bundle = get_or_fetch_bundle_by_imdb_id(str(film.tconst), use_cache=True)
    rec_tmdb = tmdb_to_display_record(bundle) if bundle else None

    display_title = film.primaryTitle
    if rec_tmdb and (rec_tmdb.get("primaryTitle") or "").strip():
        display_title = rec_tmdb["primaryTitle"].strip()

    left, right = st.columns([1, 3])
    with left:
        poster_url = _poster(rec_tmdb.get("poster_path")) if rec_tmdb else None
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            _placeholder()

    with right:
        st.subheader(display_title)
        st.caption(f"{film.startYear} • {film.genres} • {film.runtimeMinutes} min")
        st.write(f"**Note IMDb :** {film.averageRating}/10  (votes: {film.numVotes})")
        st.write(f"**Réalisateur(s) :** {film.director_names}")
        st.write(f"**Casting (top 5) :** {_dedup_cast(str(film.cast_names_top5))}")

    if rec_tmdb and rec_tmdb.get("overview_fr"):
        st.subheader("Synopsis (FR)")
        st.write(rec_tmdb["overview_fr"])
    else:
        st.caption("Synopsis FR non trouvé via TMDB pour ce film.")

    st.divider()

    raw = recommend_by_tconst(str(film.tconst), top_n=30)
    tconsts = [t for (t, _) in raw]
    ranked = _rerank(tconsts, now_ids, up_ids)[:5]

    rec_df = df[df["tconst"].isin(ranked)].copy()
    order = {t: i for i, t in enumerate(ranked)}
    rec_df["rank"] = rec_df["tconst"].map(order)
    rec_df = rec_df.sort_values("rank")

    st.subheader("Recommandations (priorité à l'affiche / bientôt)")
    cards = _enrich_local(_local_to_cards(rec_df, badge=None))
    _render_cards(cards, key_prefix="film_reco_local")


def page_demo() -> None:
    _init_state()
    df = load_movies_local()

    page = st.session_state.get("demo_page", "home")
    if page == "home":
        _page_home(df)
    elif page == "search":
        _page_search(df)
    else:
        _page_film(df)
