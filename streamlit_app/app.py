# streamlit_app/app.py

import streamlit as st

from pages_market import page_market
from pages_demo import page_demo
from pages_kpi import page_kpi

st.set_page_config(
    page_title="Projet Cinéma – Recommandation",
    layout="wide",
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Aller vers",
    [
        "Étude de marché",
        "Site démo",
        "Notes & KPI",
    ],
)

if page == "Étude de marché":
    page_market()
elif page == "Site démo":
    page_demo()
elif page == "Notes & KPI":
    page_kpi()
