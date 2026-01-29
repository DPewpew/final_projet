# final_projet/scripts/build_imdb_catalog.py
# Build:
# - data/data_processed/movies_local.csv.gz
# - data/data_processed/movies_features.csv.gz
#
# Constraints:
# - Filter Option A
# - Cast top 5 (actor/actress by ordering)
# - Directors mapped to names
# - Output files should stay under 100MB (gzip)
#
# Run from project root:
#   python scripts/build_imdb_catalog.py

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd


# -----------------------------
# Config
# -----------------------------
MIN_YEAR = 1980
MIN_VOTES = 1000
RUNTIME_MIN = 60
RUNTIME_MAX = 240
CAST_TOP_N = 5

RAW_DIR = Path("data/data_raw")
OUT_DIR = Path("data/data_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASICS_PATH = RAW_DIR / "title.basics.tsv.gz"
RATINGS_PATH = RAW_DIR / "title.ratings.tsv.gz"
CREW_PATH = RAW_DIR / "title.crew.tsv.gz"
PRINCIPALS_PATH = RAW_DIR / "title.principals.tsv.gz"
NAMES_PATH = RAW_DIR / "name.basics.tsv.gz"

OUT_LOCAL = OUT_DIR / "movies_local.csv.gz"
OUT_FEATURES = OUT_DIR / "movies_features.csv.gz"

CHUNK_BASICS = 500_000
CHUNK_RATINGS = 1_000_000
CHUNK_CREW = 1_000_000
CHUNK_PRINCIPALS = 2_000_000
CHUNK_NAMES = 1_000_000


# -----------------------------
# Helpers
# -----------------------------
def ensure_files_exist(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in data/data_raw:\n- " + "\n- ".join(missing)
        )


def to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def split_imdb_list(value: str) -> List[str]:
    """
    IMDb fields like directors can be:
    - "\\N"
    - "nm0000001,nm0000002"
    """
    if not isinstance(value, str) or value == r"\N" or value.strip() == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# -----------------------------
# Step 1: Filter title.basics (+ later join ratings)
# -----------------------------
def load_filtered_basics() -> pd.DataFrame:
    usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "isAdult",
        "startYear",
        "runtimeMinutes",
        "genres",
    ]
    dtypes = {
        "tconst": "string",
        "titleType": "string",
        "primaryTitle": "string",
        "isAdult": "Int64",
        "startYear": "string",
        "runtimeMinutes": "string",
        "genres": "string",
    }

    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        BASICS_PATH,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNK_BASICS,
        low_memory=False,
    ):
        total_rows += len(chunk)

        # Normalize types
        chunk["startYear"] = to_int_series(chunk["startYear"])
        chunk["runtimeMinutes"] = to_int_series(chunk["runtimeMinutes"])

        # Filters Option A
        mask = (
            (chunk["titleType"] == "movie")
            & (chunk["isAdult"].fillna(1) == 0)
            & (chunk["primaryTitle"].notna())
            & (chunk["primaryTitle"].str.strip() != "")
            & (chunk["startYear"].notna())
            & (chunk["startYear"] >= MIN_YEAR)
            & (chunk["runtimeMinutes"].notna())
            & (chunk["runtimeMinutes"] >= RUNTIME_MIN)
            & (chunk["runtimeMinutes"] <= RUNTIME_MAX)
            & (chunk["genres"].notna())
            & (chunk["genres"] != r"\N")
        )

        filtered = chunk.loc[mask, ["tconst", "primaryTitle", "startYear", "runtimeMinutes", "genres"]]
        chunks.append(filtered)

        print(f"[basics] scanned={total_rows:,} kept_chunk={len(filtered):,}")

    basics = pd.concat(chunks, ignore_index=True)
    basics = basics.drop_duplicates(subset=["tconst"])
    print(f"[basics] kept_total={len(basics):,}")
    return basics


# -----------------------------
# Step 2: Join ratings + filter by votes
# -----------------------------
def load_ratings_for_tconst(tconst_set: Set[str]) -> pd.DataFrame:
    usecols = ["tconst", "averageRating", "numVotes"]
    dtypes = {"tconst": "string", "averageRating": "float32", "numVotes": "Int64"}

    keep_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNK_RATINGS,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk = chunk[chunk["tconst"].isin(tconst_set)]
        if len(chunk):
            keep_chunks.append(chunk)
        print(f"[ratings] scanned={total_rows:,} matched_chunk={len(chunk):,}")

    ratings = pd.concat(keep_chunks, ignore_index=True) if keep_chunks else pd.DataFrame(columns=usecols)
    print(f"[ratings] matched_total={len(ratings):,}")
    return ratings


# -----------------------------
# Step 3: Crew directors (collect nconst)
# -----------------------------
def load_directors_for_tconst(tconst_set: Set[str]) -> pd.DataFrame:
    usecols = ["tconst", "directors"]
    dtypes = {"tconst": "string", "directors": "string"}

    rows = []
    total_rows = 0

    for chunk in pd.read_csv(
        CREW_PATH,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNK_CREW,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk = chunk[chunk["tconst"].isin(tconst_set)]
        if len(chunk):
            rows.append(chunk)
        print(f"[crew] scanned={total_rows:,} matched_chunk={len(chunk):,}")

    crew = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=usecols)
    print(f"[crew] matched_total={len(crew):,}")
    return crew


# -----------------------------
# Step 4: Principals cast top5 (collect nconst)
# -----------------------------
def load_cast_topn_for_tconst(tconst_set: Set[str]) -> pd.DataFrame:
    usecols = ["tconst", "ordering", "nconst", "category"]
    dtypes = {"tconst": "string", "ordering": "Int64", "nconst": "string", "category": "string"}

    parts = []
    total_rows = 0

    for chunk in pd.read_csv(
        PRINCIPALS_PATH,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNK_PRINCIPALS,
        low_memory=False,
    ):
        total_rows += len(chunk)

        chunk = chunk[
            (chunk["tconst"].isin(tconst_set))
            & (chunk["category"].isin(["actor", "actress"]))
            & (chunk["ordering"].notna())
            & (chunk["ordering"] <= CAST_TOP_N)
        ][["tconst", "ordering", "nconst"]]

        if len(chunk):
            parts.append(chunk)

        print(f"[principals] scanned={total_rows:,} matched_chunk={len(chunk):,}")

    principals = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["tconst", "ordering", "nconst"])
    print(f"[principals] matched_total={len(principals):,}")

    # Ensure ordering
    principals = principals.sort_values(["tconst", "ordering"], kind="mergesort")
    return principals


# -----------------------------
# Step 5: Build nconst -> primaryName mapping only for needed people
# -----------------------------
def build_name_map(needed_nconst: Set[str]) -> Dict[str, str]:
    usecols = ["nconst", "primaryName"]
    dtypes = {"nconst": "string", "primaryName": "string"}

    mapping: Dict[str, str] = {}
    total_rows = 0
    found = 0

    for chunk in pd.read_csv(
        NAMES_PATH,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNK_NAMES,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk = chunk[chunk["nconst"].isin(needed_nconst)]
        if len(chunk):
            for nconst, pname in zip(chunk["nconst"].tolist(), chunk["primaryName"].tolist()):
                if pd.notna(nconst) and pd.notna(pname):
                    mapping[str(nconst)] = str(pname)
            found += len(chunk)

        print(f"[names] scanned={total_rows:,} found_chunk={len(chunk):,} mapped={len(mapping):,}")

        # Early stop if we already mapped all needed IDs (best effort)
        if len(mapping) >= len(needed_nconst):
            break

    print(f"[names] needed={len(needed_nconst):,} mapped={len(mapping):,}")
    return mapping


# -----------------------------
# Step 6: Assemble movies_local + movies_features
# -----------------------------
def main() -> None:
    ensure_files_exist([BASICS_PATH, RATINGS_PATH, CREW_PATH, PRINCIPALS_PATH, NAMES_PATH])

    print("=== Step 1/6: Filter basics ===")
    basics = load_filtered_basics()
    tconst_set = set(basics["tconst"].astype(str).tolist())

    print("=== Step 2/6: Join ratings + filter votes ===")
    ratings = load_ratings_for_tconst(tconst_set)

    movies = basics.merge(ratings, on="tconst", how="inner")
    # votes filter
    movies = movies[movies["numVotes"].notna() & (movies["numVotes"] >= MIN_VOTES)].copy()
    movies["numVotes"] = movies["numVotes"].astype("int64")
    movies["averageRating"] = movies["averageRating"].astype("float32")

    # update tconst_set after votes filter
    tconst_set = set(movies["tconst"].astype(str).tolist())
    print(f"[catalog] after votes filter kept_total={len(movies):,}")

    print("=== Step 3/6: Directors (crew) ===")
    crew = load_directors_for_tconst(tconst_set)

    print("=== Step 4/6: Cast top 5 (principals) ===")
    principals = load_cast_topn_for_tconst(tconst_set)

    # Collect needed nconst (directors + cast)
    director_nconst: Set[str] = set()
    for s in crew["directors"].astype(str).tolist():
        director_nconst.update(split_imdb_list(s))

    cast_nconst: Set[str] = set(principals["nconst"].astype(str).tolist())
    needed_nconst = director_nconst.union(cast_nconst)

    print(f"[ids] needed_nconst_total={len(needed_nconst):,} (directors={len(director_nconst):,}, cast={len(cast_nconst):,})")

    print("=== Step 5/6: Build name map (nconst -> primaryName) ===")
    name_map = build_name_map(needed_nconst)

    # Directors: map tconst -> director_names
    def directors_to_names(directors_field: str) -> str:
        ids = split_imdb_list(directors_field)
        names = [name_map.get(i) for i in ids]
        names = [n for n in names if n]  # drop missing
        return "|".join(names)

    crew["director_names"] = crew["directors"].apply(directors_to_names)
    crew = crew[["tconst", "director_names"]]

    # Cast: map principals to names, then aggregate per tconst in ordering
    principals["actor_name"] = principals["nconst"].map(name_map).fillna("")
    principals = principals[principals["actor_name"].str.strip() != ""].copy()

    cast_agg = (
        principals.sort_values(["tconst", "ordering"], kind="mergesort")
        .groupby("tconst")["actor_name"]
        .apply(lambda s: "|".join(s.tolist()[:CAST_TOP_N]))
        .reset_index()
        .rename(columns={"actor_name": "cast_names_top5"})
    )

    print("=== Step 6/6: Build outputs ===")
    movies = movies.merge(crew, on="tconst", how="left").merge(cast_agg, on="tconst", how="left")

    # Fill missing credits with empty strings
    movies["director_names"] = movies["director_names"].fillna("")
    movies["cast_names_top5"] = movies["cast_names_top5"].fillna("")

    # Ensure final columns + dtypes
    movies = movies[
        [
            "tconst",
            "primaryTitle",
            "startYear",
            "runtimeMinutes",
            "genres",
            "averageRating",
            "numVotes",
            "director_names",
            "cast_names_top5",
        ]
    ].copy()

    movies["startYear"] = movies["startYear"].astype("int32")
    movies["runtimeMinutes"] = movies["runtimeMinutes"].astype("int32")

    # Build features (tconst + soup)
    def build_soup(row: pd.Series) -> str:
        # genres are like "Action,Drama" -> keep commas as separators
        parts = [
            str(row["genres"]).replace(",", " "),
            str(row["director_names"]).replace("|", " "),
            str(row["cast_names_top5"]).replace("|", " "),
        ]
        soup = " ".join(parts).strip().lower()
        # normalize whitespace
        soup = " ".join(soup.split())
        return soup

    features = pd.DataFrame(
        {
            "tconst": movies["tconst"].astype("string"),
            "soup": movies.apply(build_soup, axis=1).astype("string"),
        }
    )

    # Write gzipped CSV
    movies.to_csv(OUT_LOCAL, index=False, compression="gzip")
    features.to_csv(OUT_FEATURES, index=False, compression="gzip")

    print(f"[write] {OUT_LOCAL}  size={file_size_mb(OUT_LOCAL):.2f} MB")
    print(f"[write] {OUT_FEATURES}  size={file_size_mb(OUT_FEATURES):.2f} MB")
    print(f"[done] movies_local rows={len(movies):,} | movies_features rows={len(features):,}")


if __name__ == "__main__":
    main()
