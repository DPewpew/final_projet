# streamlit_app/pages_kpi.py

import streamlit as st

import pandas as pd

def page_kpi():

    # ============================
    # INTRO
    # ============================
    st.title(
        "Voici les **KPI clés** du territoire, du cinéma et du moteur de recommandation"
    )

    # ============================
    # 1) KPI INSEE
    # ============================

    st.subheader("📊 Démographie — INSEE")

    col1, col2, col3 = st.columns(3)
    col1.metric("Population totale", "118 000")
    col2.metric("60 ans et +", "47 %", "+6 pts depuis 2011")
    col3.metric("Moins de 30 ans", "22 %", "-4 pts depuis 2011")
    col1.metric("Ménages d'une personne", "41 %")
    col2.metric("Pauvreté <30 ans", "25 %")

    st.markdown("---")

    # ============================
    # 2) KPI CNC
    # ============================

    st.subheader("🎬 Cinéma — CNC")

    col1, col2, col3 = st.columns(3)
    col1.metric("Écrans (2024)", "4", "-67 % depuis 1966")
    col2.metric("Entrées annuelles", "45 000", "Stable")
    col3.metric("Entrées / habitant", "0.35", "France : 2.8")
    col1.metric("Séances annuelles", "2 000")
    col2.metric("Taux d'occupation", "0.25", "-50 % vs France")

    st.markdown("---")

    # ============================
    # 3) KPI AVANT TRAITEMENT IMDB
    # ============================

    st.header("📦 Bases de données ")

    col1, col2, col3 = st.columns(3)
    col1.metric("Films TMDB (brut)", "309 572")
    col2.metric("NB dataset", "5 fichiers")
    col3.metric("Colonnes TMDB", "40")

    
    



    # ============================
    # 4) KPI APRÈS TRAITEMENT IMDB (sans df)
    # ============================

    

    col1, col2, col3 = st.columns(3)
    col1.metric("Films IMDB (après traitement)", "38 924" )
    col2.metric( "dataset final", "1 Fichiers")
    col3.metric("Colonnes finales", "9")

    st.markdown("---")

    # ============================
    # 5) KPI TRAITEMENT IMDB (avec df)
    # ============================

   
    @st.cache_data
    def load_features():
        return pd.read_csv("data/data_processed/movies_local.csv.gz")

    df = load_features()

    processing_kpi = {
        "films_total": len(df),
        "genres_valides": df["genres"].notna().mean() * 100,
        "directors_valides": df["director_names"].notna().mean() * 100,
        "casting_valide": df["cast_names_top5"].notna().mean() * 100,
        "runtime_valide": df["runtimeMinutes"].gt(0).mean() * 100,
        "soup_completude": 100.0,
        "longueur_moyenne_soup": 55,
        "vocabulaire_tfidf": "40k–60k tokens",
    }

   


    # ============================
    # 6) KPI RECOMMANDATION
    # ============================

    st.header("🤖 Moteur de recommandation (contenu)")

    reco_kpi = {
        "films_recommandables": len(df),
        "diversite_genres": df["genres"].str.split(",").explode().nunique(),
        "richesse_cast": df["cast_names_top5"].str.split("|").explode().nunique(),
        "temps_reco": "< 50 ms",
        
    }

    col1, col2, col3 = st.columns(3)
    col1.metric("Films recommandables", "38 924")
    col2.metric("Genres uniques", f"{reco_kpi['diversite_genres']}")
    col3.metric("Acteurs uniques", f"{reco_kpi['richesse_cast']:,}")

    
    st.subheader("model choisie : Content-Based Recommender (TF-IDF + Cosine Similarity)")
    st.write("Filtrage basé sur le contenu (Content-Based Filtering)")
    st.info(
    "Le système de recommandation repose sur un filtrage basé sur le contenu. "
    "Chaque film est représenté par un vecteur TF-IDF construit à partir de ses métadonnées "
    "(genres, réalisateurs, acteurs). Les recommandations sont obtenues via une similarité cosinus."
            )
    with st.expander("variable"):
        st.code("""
                
                # Content-Based Recommender Model (TF-IDF + Cosine Similarity)
                vectorizer = TfidfVectorizer(...)

                # Similarity-based recommendation using cosine similarity
                sims = cosine_similarity(q_vec, art.matrix)
                
                """)
    st.subheader("extrait du code pour le ML")
    with st.expander("Chargement des artefacts (principe du modèle offline / online)"):
        st.code(
        """
        @st.cache_resource(show_spinner=False)
        def load_reco_artifacts() -> RecoArtifacts:
            # Chargement du vectorizer TF-IDF entraîné hors ligne
            vectorizer = joblib.load(RECO_DIR / "tfidf_vectorizer.joblib")

            # Chargement de la matrice TF-IDF contenant tous les films du catalogue
            # Chaque ligne = un film, chaque colonne = un terme
            matrix = joblib.load(RECO_DIR / "tfidf_matrix.joblib")

            # Chargement de l’index des films (tconst dans le même ordre que la matrice)
            idx = pd.read_csv(RECO_DIR / "tconst_index.csv")

            # Liste ordonnée des identifiants de films
            tconst_list = idx["tconst"].astype(str).tolist()

            # Dictionnaire pour accéder rapidement à la ligne d’un film dans la matrice
            # ex: tconst_to_row["tt0133093"] -> index de ligne
            tconst_to_row = {t: i for i, t in enumerate(tconst_list)}

            # Regroupement de tous les artefacts dans une structure unique
            return RecoArtifacts(
                vectorizer=vectorizer,
                matrix=matrix,
                tconst_list=tconst_list,
                tconst_to_row=tconst_to_row,
            )

        """
        )
    
    
    
    with st.expander("Fonction de recommandation principale (film connu)"):
        st.code(
            """ 
            def recommend_by_tconst(query_tconst: str, top_n: int = 10):
                # Chargement des artefacts TF-IDF et de la matrice
                art = load_reco_artifacts()

                # Vérification que le film existe dans le catalogue
                if query_tconst not in art.tconst_to_row:
                    return []

                # Récupération de l’index du film dans la matrice
                q_idx = art.tconst_to_row[query_tconst]

                # Vecteur TF-IDF du film cible
                q_vec = art.matrix[q_idx]

                # Calcul de la similarité cosinus entre ce film et tous les autres
                sims = cosine_similarity(q_vec, art.matrix).ravel()

                # Exclusion du film lui-même (évite l’auto-recommandation)
                sims[q_idx] = -1.0

                # Sélection des indices des top-N films les plus similaires
                top_idx = np.argpartition(-sims, top_n)[:top_n]

                # Retourne les tconst recommandés avec leur score de similarité
                return [(art.tconst_list[i], float(sims[i])) for i in top_idx]           
            
            """     
        )
    
    
    
    with st.expander("Cas film externe"):
        st.code("""
            
               def recommend_by_soup(query_soup: str, top_n: int = 10):
                    # Chargement des artefacts existants
                    art = load_reco_artifacts()

                    # Nettoyage du texte d’entrée
                    query_soup = (query_soup or "").strip().lower()
                    if not query_soup:
                        return []

                    # Transformation du texte en vecteur TF-IDF
                    # IMPORTANT : on utilise le vectorizer existant (pas de refit)
                    q_vec = art.vectorizer.transform([query_soup])

                    # Calcul de la similarité cosinus avec tous les films du catalogue
                    sims = cosine_similarity(q_vec, art.matrix).ravel()

                    # Sélection des top-N films les plus proches
                    top_idx = np.argpartition(-sims, top_n)[:top_n]

                    return [(art.tconst_list[i], float(sims[i])) for i in top_idx]
                """
            )
    
    with st.expander("Construction du TF-IDF offline"):
        st.code("""
                # Création du vectorizer TF-IDF
                vectorizer = TfidfVectorizer(
                    max_features=120_000,   # limite la taille du vocabulaire
                    ngram_range=(1, 2),     # mots seuls + paires de mots
                    min_df=2,               # ignore les termes trop rares
                    max_df=0.90             # ignore les termes trop fréquents
                )

                # Entraînement sur la colonne "soup" (représentation textuelle des films)
                X = vectorizer.fit_transform(df["soup"]) 
                """)
    
    

    st.markdown("---")

    # ============================
    # 7) APERÇU DATASET
    # ============================

    st.subheader("Aperçu du dataset après nettoyage")
    st.dataframe(df.head())



    # ============================
    # 7) info
    # ============================
    
    st.markdown("# Notes")
    st.subheader("Traitement des données & Machine Learning")
    st.markdown(
        """
    ## 1. Sources de données

    ### Données IMDB (brutes)

    * `name.basics.tsv.gz`
    * `title.basics.tsv.gz`
    * `title.crew.tsv.gz`
    * `title.principals.tsv.gz`
    * `title.ratings.tsv.gz`

    Ces fichiers constituent la base brute IMDB. Ils **ne sont jamais utilisés directement** dans Streamlit.

    ### Données externes

    * INSEE : démographie, ménages, pauvreté, salaires
    * CNC : écrans, entrées, fréquentation cinéma
    * TMDB (API) : posters, titres FR, synopsis, popularité

    ---

    ## 2. Nettoyage et préparation des données (offline)

    ### Objectif

    Construire un **catalogue films propre, léger et exploitable** pour la recommandation et l’affichage.

    ### Étapes principales

    1. **Filtrage des films**

    * Suppression des contenus non-films (séries, épisodes)
    * Filtrage temporel (films récents / pertinents)
    * Seuils de votes pour garantir une base fiable

    2. **Nettoyage des champs**

    * Harmonisation des genres
    * Suppression des valeurs manquantes critiques
    * Normalisation des titres et identifiants

    3. **Construction des fichiers finaux**

    * `movies_local.csv.gz`

        * Identifiant IMDb (`tconst`)
        * Titre principal
        * Année de sortie
        * Genres
        * Données utiles à l’affichage

    * `movies_features.csv.gz`

        * Identifiant IMDb
        * Texte descriptif ("soup") pour le ML

    👉 Ces étapes sont réalisées via des **scripts Python offline** (`scripts/`), jamais sur Streamlit Cloud.

    ---

    ## 3. Feature Engineering pour la recommandation

    ### Principe de la "soup"

    Chaque film est représenté par un texte combinant :

    * Genres
    * Réalisateur
    * Acteurs principaux

    Exemple (simplifié) :

    ```
    Drama Thriller nolan dicaprio hardy
    ```

    Ce format permet une vectorisation simple et efficace.

    ---

    ## 4. Modèle de recommandation

    ### 4.1 Logique générale du Machine Learning

    Le système de recommandation repose sur un **modèle de type Content-Based Filtering**. Le principe est de représenter chaque film sous forme vectorielle, à partir de ses caractéristiques textuelles, puis de mesurer la similarité entre films.

    L’intégralité de la phase d’apprentissage est réalisée **en amont (offline)** afin de garantir de bonnes performances dans l’application Streamlit.

    ---

    ### 4.2 Préparation des données pour le ML (en amont)

    À partir de la base nettoyée issue d’IMDB, une table dédiée au Machine Learning est construite (`movies_features.csv.gz`).

    Pour chaque film, on génère une **représentation textuelle unique appelée "soup"**, qui agrège :

    * les genres du film
    * le réalisateur principal
    * les acteurs principaux

    Exemple de soup :

    ```
    Drama Thriller nolan dicaprio hardy
    ```

    Cette étape est cruciale : elle permet de transformer des données hétérogènes (catégories, noms propres) en une forme exploitable par un modèle NLP simple.

    ---

    ### 4.3 Vectorisation TF-IDF (fit offline)

    Une fois la colonne "soup" construite pour l’ensemble du catalogue :

    1. Un **TF-IDF Vectorizer** est entraîné sur l’intégralité des soups du catalogue
    2. Chaque film est transformé en un **vecteur numérique** de dimension élevée

    Ce processus produit :

    * une matrice creuse TF-IDF (films × termes)
    * un vocabulaire pondéré par l’importance des mots

    Les fichiers générés sont :

    * `tfidf_vectorizer.joblib` → le modèle TF-IDF entraîné (fit)
    * `tfidf_matrix.joblib` → la matrice vectorisée des films
    * `tconst_index.csv` → mapping entre identifiant IMDb et index de ligne

    Ces artefacts sont sauvegardés sur disque et **ne sont jamais recalculés dans Streamlit**.

    ---

    ### 4.4 Chargement et utilisation dans Streamlit

    Dans l’application :

    * les artefacts sont chargés une seule fois via `st.cache_resource`
    * la matrice et le vectorizer restent en mémoire pour toutes les sessions

    Deux cas d’usage sont alors possibles.

    ---

    ### 4.5 Recommandation à partir d’un film du catalogue

    Lorsque l’utilisateur sélectionne un film déjà présent dans la base locale :

    1. On récupère son index dans la matrice TF-IDF
    2. On calcule la **similarité cosinus** entre son vecteur et tous les autres films
    3. On trie les scores et on retourne les films les plus similaires

    Ce mécanisme est rapide car il s’appuie uniquement sur des calculs matriciels en mémoire.

    ---

    ### 4.6 Recommandation à partir d’un film externe (API TMDB)

    Pour un film **absent du catalogue local** (par exemple issu de la recherche TMDB) :

    1. Les informations du film sont récupérées via l’API TMDB
    2. Une soup est construite dynamiquement (genres + réalisateur + acteurs)
    3. Cette soup est **transformée** avec le vectorizer existant (pas de refit)
    4. Le vecteur obtenu est comparé à la matrice TF-IDF locale via similarité cosinus

    👉 Le modèle n’est jamais réentraîné : on applique uniquement une **transformation** cohérente avec l’apprentissage initial.

    ---

    ### 4.7 Cohérence entre données locales et données API

    Le point clé du système est la **cohérence du pipeline** :

    * même logique de soup
    * même normalisation (minuscules, espaces)
    * même vectorizer

    Cela garantit que les films issus de l’API TMDB sont projetés dans **le même espace vectoriel** que les films du catalogue local.

    ---

    ### 4.8 Pourquoi ce choix de modèle

    Ce modèle a été choisi car il :

    * est explicable
    * ne nécessite pas de données utilisateurs
    * est rapide et robuste
    * est parfaitement adapté à un contexte Data Analyst

    Il permet de démontrer une chaîne ML complète sans complexité inutile.

    ---

    ### 4.9 Limites spécifiques du ML

    * Pas de prise en compte des préférences utilisateurs
    * Sensible à la qualité des métadonnées (genres, casting)
    * Ne capture pas les relations sémantiques profondes

    ---

    ### 4.10 Évolutions possibles

    * Passage à des embeddings (Word2Vec, SBERT)
    * Ajout d’un scoring hybride (contenu + popularité)
    * Intégration de feedback utilisateur

    ### Type de modèle

    * **Content-Based Filtering**
    * Aucun apprentissage supervisé
    * Pas de données utilisateur

    ### Méthode

    1. Vectorisation TF-IDF sur la soup
    2. Calcul de similarité cosinus entre films
    3. Classement des films les plus proches

    ### Artefacts produits

    * `tfidf_vectorizer.joblib`
    * `tfidf_matrix.joblib`
    * `tconst_index.csv`

    Ces fichiers sont chargés **une seule fois** dans Streamlit grâce à `st.cache_resource`.

    ---

    ## 5. Intégration TMDB (enrichissement)

    ### Rôle de TMDB

    * Titres en français
    * Posters et backdrops
    * Synopsis
    * Popularité

    ### Fonctionnement

    * Appels API encapsulés dans `tmdb_client.py`
    * Cache disque + cache Streamlit
    * Aucun enrichissement massif au chargement

    ### Principe clé

    👉 **L’enrichissement se fait uniquement à l’affichage** (Top 5 / cartes visibles)

    Cela garantit :

    * Performance
    * Respect des quotas API

    ---

    ## 6. Reranking et contextualisation

    Pour certains cas (films à l’affiche / à venir) :

    * Construction de sets IMDb `now_playing` / `upcoming`
    * Permet d’annoter ou prioriser les recommandations

    Ces calculs sont :

    * Mis en cache
    * Recalculés à intervalle contrôlé (TTL)

    ---

    ## 7. Architecture Streamlit (résumé)

    * **Offline** : nettoyage, feature engineering, ML
    * **Online (Streamlit)** :

    * Chargement des fichiers finaux
    * Recommandation en temps réel
    * Enrichissement visuel à la demande

    Cette séparation garantit :

    * Performance
    * Reproductibilité
    * Scalabilité

    ---

    ## 8. Limites identifiées

    * Pas de personnalisation utilisateur
    * Recommandation basée uniquement sur le contenu
    * Dépendance partielle à une API externe (TMDB)

    ---

    ## Conclusion

    Le projet met en œuvre une **chaîne complète de data analysis appliquée** :

    * collecte
    * nettoyage
    * feature engineering
    * machine learning
    * déploiement applicatif

    Le tout dans une architecture **adaptée à un contexte Data Analyst**, claire, performante et justifiable.


            """
        )
    st.markdown(
            """
            # 📊 KPI – Traitement des données

            """        
        )