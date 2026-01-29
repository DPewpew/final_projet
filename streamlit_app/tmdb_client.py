# streamlit_app/tmdb_client.py
# TMDB client for Streamlit (v3 API key).
#
# Features:
# - Reads TMDB API key from .streamlit/secrets.toml (TMDB_API_KEY)
# - Disk cache (JSON) to avoid re-hitting API
# - Now Playing / Upcoming (FR)
# - Details (FR), Credits (cast + director), External IDs (imdb_id)
# - Helpers to build a "soup" compatible with your ML (genres + director + cast_top5)
#
# Usage (in Streamlit):
#   from tmdb_client import (
#       get_now_playing_fr, get_upcoming_fr, get_tmdb_details_bundle,
#       build_soup_from_tmdb_bundle, tmdb_to_display_record
#   )
#
# Notes:
# - Do NOT commit secrets.toml (add to .gitignore)
# - Streamlit Cloud: set TMDB_API_KEY in secrets UI

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
TMDB_BASE = "https://api.themoviedb.org/3"
LANG_FR = "fr-FR"
REGION_FR = "FR"

# Disk cache location (committable; contains NO secret)
BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "data" / "data_processed" / "tmdb_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_NOW_PLAYING = CACHE_DIR / "now_playing_fr.json"
CACHE_UPCOMING = CACHE_DIR / "upcoming_fr.json"
CACHE_BUNDLES = CACHE_DIR / "bundles_by_tmdb_id.json"  # details+credits+external_ids by tmdb_id

# TTL in seconds
TTL_LISTS = 6 * 3600          # 6 hours for now/upcoming
TTL_BUNDLES = 30 * 24 * 3600  # 30 days for movie bundles (stable)

TTL_NOW_PLAYING = 6 * 3600
TTL_UPCOMING = 24 * 3600
TTL_SEARCH = 2 * 3600
TTL_PERSON_CREDITS = 7 * 24 * 3600
TTL_FIND_IMDB = 30 * 24 * 3600


# -----------------------------
# Exceptions
# -----------------------------
class TMDBError(RuntimeError):
    pass


# -----------------------------
# Secrets / HTTP helpers
# -----------------------------
def _get_api_key() -> str:
    key = st.secrets.get("TMDB_API_KEY", "")
    if not key:
        raise TMDBError(
            "TMDB_API_KEY missing. Add it to .streamlit/secrets.toml or Streamlit Cloud Secrets."
        )
    return str(key)


def _request_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform a GET request to TMDB with basic error handling.
    """
    api_key = _get_api_key()
    url = f"{TMDB_BASE}{path}"
    p = dict(params or {})
    p["api_key"] = api_key

    r = requests.get(url, params=p, timeout=20)
    if r.status_code != 200:
        # Try to extract message
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise TMDBError(f"TMDB request failed ({r.status_code}) for {path}: {msg}")

    return r.json()


# -----------------------------
# Disk cache helpers
# -----------------------------
def _read_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_fresh(ts: float, ttl: int) -> bool:
    return (time.time() - ts) <= ttl

def _bundle_path_for_tmdb_id(tmdb_id: int) -> Path:
    return CACHE_DIR / f"bundle_tmdb_{int(tmdb_id)}.json"


def _read_bundle_file(tmdb_id: int) -> Optional[Dict[str, Any]]:
    p = _bundle_path_for_tmdb_id(tmdb_id)
    data = _read_cache(p)
    if not data:
        return None
    if not _is_fresh(float(data.get("_ts", 0)), TTL_BUNDLES):
        return None
    return data


def _write_bundle_file(tmdb_id: int, bundle: Dict[str, Any]) -> None:
    p = _bundle_path_for_tmdb_id(tmdb_id)
    _write_cache(p, bundle)


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class TMDBListItem:
    tmdb_id: int
    title: str
    overview: str
    poster_path: Optional[str]
    backdrop_path: Optional[str]
    release_date: Optional[str]
    vote_average: Optional[float]
    vote_count: Optional[int]
    popularity: Optional[float]


@dataclass(frozen=True)
class TMDBBundle:
    tmdb_id: int
    details: Dict[str, Any]
    credits: Dict[str, Any]
    external_ids: Dict[str, Any]


# -----------------------------
# Public: Now Playing / Upcoming
# -----------------------------
def _parse_list_results(payload: Dict[str, Any]) -> List[TMDBListItem]:
    items: List[TMDBListItem] = []
    for r in payload.get("results", []):
        items.append(
            TMDBListItem(
                tmdb_id=int(r.get("id")),
                title=str(r.get("title") or r.get("name") or ""),
                overview=str(r.get("overview") or ""),
                poster_path=r.get("poster_path"),
                backdrop_path=r.get("backdrop_path"),
                release_date=r.get("release_date"),
                vote_average=r.get("vote_average"),
                vote_count=r.get("vote_count"),
                popularity=r.get("popularity"),
            )
        )
    return items

@st.cache_data(ttl=TTL_NOW_PLAYING, show_spinner=False)
def get_now_playing_fr(pages: int = 1, use_cache: bool = True) -> List[TMDBListItem]:
    """
    Fetch now playing movies in FR language/region.
    pages=1 usually enough (20 items). Increase if needed.
    """
    if use_cache:
        cached = _read_cache(CACHE_NOW_PLAYING)
        if cached and _is_fresh(cached.get("_ts", 0), TTL_NOW_PLAYING):
            return _parse_list_results(cached["payload"])

    # Fetch pages and merge results
    merged_payload: Dict[str, Any] = {"results": []}
    for page in range(1, max(1, pages) + 1):
        payload = _request_json(
            "/movie/now_playing",
            params={"language": LANG_FR, "region": REGION_FR, "page": page},
        )
        merged_payload["results"].extend(payload.get("results", []))

    if use_cache:
        _write_cache(CACHE_NOW_PLAYING, {"_ts": time.time(), "payload": merged_payload})

    return _parse_list_results(merged_payload)

@st.cache_data(ttl=TTL_UPCOMING, show_spinner=False)
def get_upcoming_fr(pages: int = 1, use_cache: bool = True) -> List[TMDBListItem]:
    """
    Fetch upcoming movies in FR language/region.
    """
    if use_cache:
        cached = _read_cache(CACHE_UPCOMING)
        if cached and _is_fresh(cached.get("_ts", 0), TTL_UPCOMING):
            return _parse_list_results(cached["payload"])

    merged_payload: Dict[str, Any] = {"results": []}
    for page in range(1, max(1, pages) + 1):
        payload = _request_json(
            "/movie/upcoming",
            params={"language": LANG_FR, "region": REGION_FR, "page": page},
        )
        merged_payload["results"].extend(payload.get("results", []))

    if use_cache:
        _write_cache(CACHE_UPCOMING, {"_ts": time.time(), "payload": merged_payload})

    return _parse_list_results(merged_payload)


# -----------------------------
# Public: Details / Credits / External IDs (cached bundle)
# -----------------------------
def _load_bundles_store() -> Dict[str, Any]:
    store = _read_cache(CACHE_BUNDLES)
    if not store or "items" not in store:
        return {"_ts": time.time(), "items": {}}
    return store


def _save_bundles_store(store: Dict[str, Any]) -> None:
    store["_ts"] = time.time()
    _write_cache(CACHE_BUNDLES, store)


def get_movie_details_fr(tmdb_id: int) -> Dict[str, Any]:
    return _request_json(f"/movie/{tmdb_id}", params={"language": LANG_FR})


def get_movie_credits(tmdb_id: int) -> Dict[str, Any]:
    return _request_json(f"/movie/{tmdb_id}/credits", params={})


def get_movie_external_ids(tmdb_id: int) -> Dict[str, Any]:
    return _request_json(f"/movie/{tmdb_id}/external_ids", params={})


@st.cache_data(ttl=TTL_BUNDLES, show_spinner=False)
def get_tmdb_details_bundle(tmdb_id: int, use_cache: bool = True) -> TMDBBundle:
    tmdb_id = int(tmdb_id)

    # 1) Cache disque "1 fichier par film"
    if use_cache:
        cached_file = _read_bundle_file(tmdb_id)
        if cached_file:
            return TMDBBundle(
                tmdb_id=tmdb_id,
                details=cached_file["details"],
                credits=cached_file["credits"],
                external_ids=cached_file["external_ids"],
            )

    # 2) Fallback: ancien store global
    tmdb_id_str = str(tmdb_id)
    if use_cache:
        store = _load_bundles_store()
        item = store["items"].get(tmdb_id_str)
        if item and _is_fresh(item.get("_ts", 0), TTL_BUNDLES):
            # écriture opportuniste en fichier par film
            _write_bundle_file(tmdb_id, item)
            return TMDBBundle(
                tmdb_id=tmdb_id,
                details=item["details"],
                credits=item["credits"],
                external_ids=item["external_ids"],
            )

    details = get_movie_details_fr(tmdb_id)
    credits = get_movie_credits(tmdb_id)
    external_ids = get_movie_external_ids(tmdb_id)

    bundle = {"_ts": time.time(), "details": details, "credits": credits, "external_ids": external_ids}

    if use_cache:
        # save both (file + global store for backward compatibility)
        _write_bundle_file(tmdb_id, bundle)
        store = _load_bundles_store()
        store["items"][tmdb_id_str] = bundle
        _save_bundles_store(store)

    return TMDBBundle(tmdb_id=tmdb_id, details=details, credits=credits, external_ids=external_ids)



# -----------------------------
# Helpers: Build soup + mapping + display record
# -----------------------------
def _extract_genre_names(details: Dict[str, Any]) -> List[str]:
    genres = details.get("genres") or []
    out = []
    for g in genres:
        name = g.get("name")
        if name:
            out.append(str(name))
    return out


def _extract_director_name(credits: Dict[str, Any]) -> str:
    crew = credits.get("crew") or []
    # First director
    for c in crew:
        if str(c.get("job", "")).lower() == "director":
            return str(c.get("name", "")).strip()
    return ""


def _extract_cast_top_n(credits: Dict[str, Any], n: int = 5) -> List[str]:
    cast = credits.get("cast") or []
    names: List[str] = []
    for c in cast[: max(0, n)]:
        nm = str(c.get("name", "")).strip()
        if nm:
            names.append(nm)
    return names


def build_soup_from_tmdb_bundle(bundle: TMDBBundle, cast_top_n: int = 5) -> str:
    """
    Build a soup compatible with your TF-IDF (genres + director + cast_top5).
    Normalize like your local build (lowercase, whitespace normalized).
    """
    genre_names = _extract_genre_names(bundle.details)
    director = _extract_director_name(bundle.credits)
    cast = _extract_cast_top_n(bundle.credits, n=cast_top_n)

    parts = [
        " ".join(genre_names).replace(",", " "),
        director.replace("|", " "),
        " ".join(cast).replace("|", " "),
    ]
    soup = " ".join(parts).strip().lower()
    soup = " ".join(soup.split())
    return soup


def tmdb_bundle_to_imdb_id(bundle: TMDBBundle) -> str:
    """
    Returns imdb_id like 'tt1234567' or '' if missing.
    """
    imdb_id = bundle.external_ids.get("imdb_id")
    return str(imdb_id).strip() if imdb_id else ""


def tmdb_to_display_record(bundle: TMDBBundle) -> Dict[str, Any]:
    """
    Create a display-friendly record aligned with your local schema as much as possible.
    (For UI display; NOT for ML fit.)
    """
    details = bundle.details
    credits = bundle.credits

    title = str(details.get("title") or "").strip()
    release_date = details.get("release_date") or ""
    year = int(release_date[:4]) if isinstance(release_date, str) and len(release_date) >= 4 else None

    genres = _extract_genre_names(details)
    director = _extract_director_name(credits)
    cast = _extract_cast_top_n(credits, n=5)

    return {
        "tmdb_id": bundle.tmdb_id,
        "imdb_id": tmdb_bundle_to_imdb_id(bundle),  # may be ""
        "primaryTitle": title,
        "startYear": year,
        "runtimeMinutes": details.get("runtime"),
        "genres": ",".join(genres) if genres else "",
        "overview_fr": str(details.get("overview") or "").strip(),
        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
        "director_names": director,
        "cast_names_top5": "|".join(cast) if cast else "",
        # TMDB scoring (keep separate from IMDb)
        "tmdb_vote_average": details.get("vote_average"),
        "tmdb_vote_count": details.get("vote_count"),
        "popularity": details.get("popularity"),
        "release_date": release_date,
        "status": details.get("status"),
    }

@st.cache_data(ttl=TTL_NOW_PLAYING, show_spinner=False)
def build_now_upcoming_imdb_id_sets(pages_now: int = 1, pages_upcoming: int = 1) -> Tuple[set, set]:
    """
    Builds sets of imdb_id (tt...) for now_playing and upcoming.
    This requires external_ids per movie => done with bundle calls (cached).
    Use with care: pages=1 is usually OK.
    """
    now_items = get_now_playing_fr(pages=pages_now, use_cache=True)
    up_items = get_upcoming_fr(pages=pages_upcoming, use_cache=True)

    now_imdb_ids = set()
    up_imdb_ids = set()

    for it in now_items:
        b = get_tmdb_details_bundle(it.tmdb_id, use_cache=True)
        imdb_id = tmdb_bundle_to_imdb_id(b)
        if imdb_id:
            now_imdb_ids.add(imdb_id)

    for it in up_items:
        b = get_tmdb_details_bundle(it.tmdb_id, use_cache=True)
        imdb_id = tmdb_bundle_to_imdb_id(b)
        if imdb_id:
            up_imdb_ids.add(imdb_id)

    return now_imdb_ids, up_imdb_ids

@st.cache_data(ttl=TTL_FIND_IMDB, show_spinner=False)
def find_tmdb_id_by_imdb_id(imdb_id: str) -> Optional[int]:
    """
    Map an IMDb id (tt...) to a TMDB movie id using /find.
    Returns tmdb_id or None.
    """
    imdb_id = (imdb_id or "").strip()
    if not imdb_id:
        return None

    payload = _request_json(
        f"/find/{imdb_id}",
        params={"external_source": "imdb_id", "language": LANG_FR},
    )
    results = payload.get("movie_results") or []
    if not results:
        return None
    return int(results[0].get("id"))

@st.cache_data(ttl=TTL_BUNDLES, show_spinner=False)
def get_or_fetch_bundle_by_imdb_id(imdb_id: str, use_cache: bool = True) -> Optional[TMDBBundle]:
    """
    Convenience helper: imdb_id -> tmdb_id -> bundle
    """
    tmdb_id = find_tmdb_id_by_imdb_id(imdb_id)
    if not tmdb_id:
        return None
    return get_tmdb_details_bundle(tmdb_id, use_cache=use_cache)

@st.cache_data(ttl=TTL_SEARCH, show_spinner=False)
def search_movie_fr(query: str, pages: int = 1) -> List[TMDBListItem]:
    query = (query or "").strip()
    if not query:
        return []
    merged_payload: Dict[str, Any] = {"results": []}
    for page in range(1, max(1, pages) + 1):
        payload = _request_json(
            "/search/movie",
            params={"language": LANG_FR, "region": REGION_FR, "query": query, "page": page, "include_adult": False},
        )
        merged_payload["results"].extend(payload.get("results", []))
    return _parse_list_results(merged_payload)

@st.cache_data(ttl=TTL_SEARCH, show_spinner=False)
def search_person_fr(query: str, pages: int = 1) -> List[Dict[str, Any]]:
    """
    Search a person, returns raw TMDB payload items (need credit lookup after).
    """
    query = (query or "").strip()
    if not query:
        return []
    merged: List[Dict[str, Any]] = []
    for page in range(1, max(1, pages) + 1):
        payload = _request_json(
            "/search/person",
            params={"language": LANG_FR, "query": query, "page": page, "include_adult": False},
        )
        merged.extend(payload.get("results", []))
    return merged

@st.cache_data(ttl=TTL_PERSON_CREDITS, show_spinner=False)
def get_person_movie_credits(person_id: int) -> List[TMDBListItem]:
    payload = _request_json(f"/person/{person_id}/movie_credits", params={"language": LANG_FR})
    # payload contains "cast" + "crew". For actor search: use "cast"
    items: List[TMDBListItem] = []
    for r in payload.get("cast", [])[:200]:
        items.append(
            TMDBListItem(
                tmdb_id=int(r.get("id")),
                title=str(r.get("title") or ""),
                overview=str(r.get("overview") or ""),
                poster_path=r.get("poster_path"),
                backdrop_path=r.get("backdrop_path"),
                release_date=r.get("release_date"),
                vote_average=r.get("vote_average"),
                vote_count=r.get("vote_count"),
                popularity=r.get("popularity"),
            )
        )
    return items
