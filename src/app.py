# app.py
import re
from functools import lru_cache
import os
import pandas as pd
import streamlit as st
import csv
import io
import requests
import numpy as np

from datetime import datetime
from email.utils import parsedate_to_datetime


# ---------- CONFIG ----------
st.set_page_config(page_title="Immo Dashboard", layout="wide")

DEFAULT_CSV_URL = "https://raw.githubusercontent.com/MarylineFONTA/PipeLine-Immobilier/refs/heads/main/data/cleaned_data.csv"

PARIS_ARR_COORDS = {
    "75001": (48.8625, 2.3369), "75002": (48.8686, 2.3412), "75003": (48.8627, 2.3601),
    "75004": (48.8544, 2.3570), "75005": (48.8430, 2.3500), "75006": (48.8494, 2.3317),
    "75007": (48.8567, 2.3125), "75008": (48.8748, 2.3170), "75009": (48.8761, 2.3378),
    "75010": (48.8786, 2.3590), "75011": (48.8570, 2.3760), "75012": (48.8333, 2.4022),
    "75013": (48.8270, 2.3550), "75014": (48.8322, 2.3230), "75015": (48.8417, 2.2986),
    "75016": (48.8625, 2.2681), "75116": (48.8666, 2.2699),
    "75017": (48.8850, 2.3090), "75018": (48.8920, 2.3440), "75019": (48.8890, 2.3830),
    "75020": (48.8640, 2.3980),
}

# ---------- UTILS ----------

@st.cache_data(ttl=600)
def get_csv_last_modified(url: str) -> datetime | None:
    """Renvoie la date locale du dernier commit qui a modifié le fichier pointé
    par une URL GitHub (raw ou blob). Ne plante pas si st.secrets est absent."""
    def to_local(dt: datetime) -> datetime:
        try:
            from zoneinfo import ZoneInfo
            return dt.astimezone(ZoneInfo("Europe/Paris"))
        except Exception:
            return dt.astimezone()

    def parse_github(u: str):
        # -> (owner, repo, branch, path) ou None
        if "raw.githubusercontent.com" in u:
            parts = u.split("raw.githubusercontent.com/")[-1].split("/")
            owner, repo = parts[0], parts[1]
            if len(parts) >= 5 and parts[2] == "refs" and parts[3] == "heads":
                branch = parts[4]
                path = "/".join(parts[5:])
            else:
                branch = parts[2]
                path = "/".join(parts[3:])
            return owner, repo, branch, path
        if "github.com" in u and "/blob/" in u:
            tail = u.split("github.com/")[-1]
            owner, repo, _, branch, *pp = tail.split("/")
            path = "/".join(pp)
            return owner, repo, branch, path
        return None

    parsed = parse_github(url)
    if not parsed:
        # URL non GitHub : dernier recours = en-tête Last-Modified (souvent absent)
        try:
            r = requests.head(url, timeout=10, allow_redirects=True,
                              headers={"User-Agent": "immo-dashboard"})
            lm = r.headers.get("Last-Modified") or r.headers.get("last-modified")
            if lm:
                from email.utils import parsedate_to_datetime
                return to_local(parsedate_to_datetime(lm))
        except Exception:
            pass
        return None

    owner, repo, branch, path = parsed
    api = "https://api.github.com/repos/{}/{}/commits".format(owner, repo)
    params = {"path": path, "sha": branch, "per_page": 1}
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "immo-dashboard"}

    # 🔐 Token optionnel : d'abord variable d'env, sinon secrets si dispo
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        try:
            token = st.secrets.get("GITHUB_TOKEN", None)  # ne plante pas si secrets absent
        except Exception:
            token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.get(api, params=params, headers=headers, timeout=15)
        if r.ok and r.json():
            iso = r.json()[0]["commit"]["committer"]["date"]  # "YYYY-MM-DDTHH:MM:SSZ"
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return to_local(dt)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=True, ttl=600)
def load_csv(url: str) -> pd.DataFrame:
    # Convertir URL GitHub "blob" -> "raw"
    if url.startswith("http") and "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

    # Récupérer le contenu (pour détecter une éventuelle 1re ligne 'sep=;')
    if url.startswith("http"):
        text = requests.get(url, timeout=30).text
    else:
        with open(url, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    # Sauter la ligne 'sep=;' éventuelle
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("sep="):
        text = "\n".join(lines[1:])

    # Lecture robuste en supposant que ton pipeline écrit avec ';' et des guillemets "
    buf = io.StringIO(text)
    return pd.read_csv(
        buf,
        sep=";",
        engine="python",
        encoding="utf-8",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="error",  # mets "warn" pour localiser si besoin
    )

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Harmonise les noms de colonnes usuels
    cols = {c.lower(): c for c in df.columns}
    rename = {}

    def has(*names): return next((cols[n] for n in names if n in cols), None)

    c_price_eur   = has("prix_eur", "price_eur")
    c_surface_m2 = has( "surface_m2", "surface_m2 (m2)")
    c_addr    = has("address", "adresse", "location")
    c_cp      = has("postal_code", "cp", "code_postal")
    c_lat     = has("lat", "latitude", "y")
    c_lon     = has("lon", "lng", "longitude", "x")
    c_url     = has("url", "lien")

    if c_price_eur:   rename[c_price_eur]   = "price_eur"
    if c_surface_m2: rename[c_surface_m2] = "surface_m2"
    if c_addr:    rename[c_addr]    = "address"
    if c_cp:      rename[c_cp]      = "postal_code"
    if c_lat:     rename[c_lat]     = "lat"
    if c_lon:     rename[c_lon]     = "lon"
    if c_url:     rename[c_url]     = "url"

    df = df.rename(columns=rename)

    # Types
    if "price_eur" in df:
        df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")
    if "surface_m2" in df:
        df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")

    # Ajouts utiles
    '''if {"price_eur", "surface_m2"}.issubset(df.columns):
     df["eur_m2"] = (df["price_eur"] / df["surface_m2"]).round(0)
  ''' 
    # Ville (simple extraction depuis l’adresse)
    if "address" in df and "city" not in df.columns:
        df["city"] = df["address"].fillna("").apply(extract_city)

    # Nettoyage de base
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])
    return df

def extract_city(addr: str) -> str | None:
    if not addr:
        return None
    # Exemples : "Paris 14ème (75014)" / "Lyon (69003)" / "Montpellier (34000)"
    m = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ'’\- ]+)\s*(?:\d+(?:er|e|ème)?)?\s*\(\d{5}\)", addr)
    if m: 
        return m.group(1).strip()
    # fallback très simple : avant la parenthèse
    m = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ'’\- ]+)\s*\(", addr)
    return m.group(1).strip() if m else None

@lru_cache(maxsize=2048)
def geocode_address(addr: str) -> tuple[float | None, float | None]:
    """Géocodage Nominatim + normalisation Paris arrondissements."""
    if not addr:
        return (None, None)

    try:
        from geopy.geocoders import Nominatim
        import time, re

        geolocator = Nominatim(user_agent="streamlit-immo-dashboard")
        q = addr.strip()

        # 1) Si CP déjà présent (ex: 75014), on s'en sert tel quel
        m_cp = re.search(r"\b(\d{5})\b", q)
        if m_cp:
            cp = m_cp.group(1)
            # on renforce la requête
            q1 = f"{q}, {cp}, France"
            loc = geolocator.geocode(q1, exactly_one=True, addressdetails=False, country_codes="fr", timeout=10)
            if loc:
                return (loc.latitude, loc.longitude)

        # 2) Si mention "Paris xx(e/ème)" sans CP, on fabrique le CP
        m_arr = re.search(r"paris[^0-9]*(\d{1,2})(?:er|e|ème)?", q, re.IGNORECASE)
        if m_arr:
            n = int(m_arr.group(1))
            cps = []
            if 1 <= n <= 20:
                cps.append(f"750{n:02d}")
                # cas particulier du 16e : parfois 75116
                if n == 16:
                    cps.append("75116")
            # On tente avec les CP construits
            for cp_try in cps:
                q2 = f"{q}, {cp_try}, Paris, France"
                loc = geolocator.geocode(q2, exactly_one=True, addressdetails=False, country_codes="fr", timeout=10)
                if loc:
                    return (loc.latitude, loc.longitude)
                time.sleep(1)  # respecter ~1 req/s

        # 3) Fallback : préciser la ville/pays
        if "paris" in q.lower():
            q3 = f"{q}, Paris, France"
        else:
            q3 = f"{q}, France"

        loc = geolocator.geocode(q3, exactly_one=True, addressdetails=False, country_codes="fr", timeout=10)
        if loc:
            return (loc.latitude, loc.longitude)

    except Exception:
        pass

    return (None, None)


# ---------- SIDEBAR ----------
st.sidebar.header("Paramètres")
csv_url = st.sidebar.text_input("URL CSV (GitHub raw)", value=DEFAULT_CSV_URL, help="https://github.com/MarylineFONTA/PipeLine-Immobilier/blob/main/data/cleaned_data.csv")

if st.sidebar.button("↻ Recharger les données"):
    load_csv.clear()   # vide le cache
    st.rerun()
    
df = load_csv(csv_url)


st.sidebar.markdown("### Filtres")
price_eur_min, price_eur_max = (
    int(df["price_eur"].min()) if "price_eur" in df and df["price_eur"].notna().any() else 0,
    int(df["price_eur"].max()) if "price_eur" in df and df["price_eur"].notna().any() else 1_000_000,
)
surface_m2_min, surface_m2_max = (
    int(df["surface_m2"].min()) if "surface_m2" in df and df["surface_m2"].notna().any() else 0,
    int(df["surface_m2"].max()) if "surface_m2" in df and df["surface_m2"].notna().any() else 200,
)

def fmt_fr(n): return f"{int(n):,}".replace(",", " ")

step = int(max(1000, (price_eur_max - price_eur_min)//100 or 1))
options = [price_eur_min]
cur = price_eur_min + step
while cur < price_eur_max:
    options.append(cur)
    cur += step
options.append(price_eur_max)  # dernier = vrai max

price_eur_sel = st.sidebar.select_slider(
    "Prix (€)",
    options=options,
    value=(options[0], options[-1]),
    format_func=lambda x: f"{fmt_fr(x)} €",
    key=f"price_{options[0]}_{options[-1]}",
)


surface_m2_sel = st.sidebar.slider("Surface (m²)", min_value=surface_m2_min, max_value=surface_m2_max,
                                value=(surface_m2_min, surface_m2_max), step=1)

cities = sorted([c for c in df.get("city", pd.Series([])).dropna().unique().tolist()])
city_sel = st.sidebar.multiselect("Ville", cities, default=[])

q = st.sidebar.text_input("Recherche texte (dans l’adresse)", value="")

do_geocode = False

#do_geocode = st.sidebar
# checkbox("Géocoder les lignes sans lat/lon (Nominatim)", value=False,
#                                help="À utiliser avec parcimonie (quotas). Le résultat est mis en cache en mémoire.")

# ---------- MAIN ----------
#st.title("🏠 Tableau de bord immobilier - Paris (Source : DVF.com)")
st.markdown(
    """
    <style>
        /* Réduit l'espace blanc au-dessus du contenu principal */
        .block-container {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- juste avant le header ---
last_dt  = get_csv_last_modified(csv_url)               # ta fonction déjà définie
last_txt = last_dt.strftime("%d/%m/%Y %H:%M") if last_dt else "indisponible"

# --- header avec Source + Date sur UNE seule ligne ---
st.markdown(
    f"""
    <h1 style='text-align:left; font-size:38px; color:#2C3E50;'>
        🏠 Tableau de bord immobilier - Paris
    </h1>
    <p style='text-align:left; font-size:16px; color:gray; margin-top:-10px;'>
        (Source : DVF.com&nbsp;•&nbsp;🗓️ Données mises à jour : {last_txt})
    </p>
    """,
    unsafe_allow_html=True
)



# KPIs en haut
left, mid, right = st.columns(3)
left.metric("Annonces (total)", len(df))
if "price_eur" in df:
    mid.metric("Prix moyen (global)", f"{int(df['price_eur'].mean(skipna=True)):,} €".replace(",", " "))
if {"price_eur", "surface_m2"}.issubset(df.columns):
    right.metric("€/m² moyen (global)", f"{int(df['price_per_m2'].mean(skipna=True)):,} €".replace(",", " "))

# Filtres
mask = pd.Series(True, index=df.index)
if "price_eur" in df:
    mask &= df["price_eur"].between(price_eur_sel[0], price_eur_sel[1], inclusive="both")
if "surface_m2" in df:
    mask &= df["surface_m2"].between(surface_m2_sel[0], surface_m2_sel[1], inclusive="both")
if city_sel and "city" in df:
    mask &= df["city"].isin(city_sel)
if q:
    if "address" in df:
        mask &= df["address"].str.contains(q, case=False, na=False)
    elif "city" in df:
        mask &= df["city"].str.contains(q, case=False, na=False)

dff = df.loc[mask].copy()

# KPIs filtrés
st.subheader("🔎 Résultats filtrés")
k1, k2, k3 = st.columns(3)
k1.metric("Annonces retenues", len(dff))
if "price_eur" in dff and len(dff):
    k2.metric("Prix moyen (filtré)", f"{int(dff['price_eur'].mean(skipna=True)):,} €".replace(",", " "))
if {"price_eur", "surface_m2"}.issubset(dff.columns) and len(dff):
    k3.metric("€/m² moyen (filtré)", f"{int(dff['price_per_m2'].mean(skipna=True)):,} €".replace(",", " "))

import numpy as np
import altair as alt

if "price_eur" in dff and dff["price_eur"].notna().any():
    st.markdown("### 📈 Histogramme des prix (5 classes)")
    s = pd.to_numeric(dff["price_eur"], errors="coerce").dropna()
    if len(s) == 0:
        st.info("Pas de prix exploitables pour l’histogramme.")
    else:
        lo, hi = float(s.min()), float(s.max())
        if np.isclose(lo, hi):
            edges = np.array([lo - 0.5, hi + 0.5])  # un seul bin visuel
        else:
            edges = np.linspace(lo, hi, 6)          # 5 classes

        cats = pd.cut(s, bins=edges, include_lowest=True, right=True)
        counts = cats.value_counts(sort=False)

        labels = [f"{int(edges[i]):,} – {int(edges[i+1]):,} €".replace(",", " ")
                  for i in range(len(edges)-1)]

        chart_df = pd.DataFrame({
            "Intervalle": pd.Categorical(labels, categories=labels, ordered=True),
            "Nombre": counts.values
        })

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Intervalle:N", sort=labels, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("Nombre:Q", axis=alt.Axis(title=None)),
            tooltip=["Intervalle", "Nombre"]
        ).properties(height=320)

        st.altair_chart(chart, use_container_width=True)


    # Affichage (ordre forcé)
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Intervalle:N", sort=labels),
        y="Nombre:Q",
        tooltip=["Intervalle", "Nombre"]
    ).properties(height=300)


    # Si tu préfères st.bar_chart :
    # st.bar_chart(chart_df.set_index("Intervalle")["Nombre"])



# -------------------- CARTE --------------------
import pydeck as pdk

# 0) Colonnes coordonnées toujours présentes et numériques
dff["lat"] = pd.to_numeric(dff.get("lat", pd.Series(pd.NA, index=dff.index)), errors="coerce")
dff["lon"] = pd.to_numeric(dff.get("lon", pd.Series(pd.NA, index=dff.index)), errors="coerce")

def _try_show_map(df):
    has_coords = df[["lat", "lon"]].notna().all(axis=1)
    if not has_coords.any():
        return False

    map_df = df.loc[has_coords, ["lat", "lon", "address", "price_eur", "url"]].copy()
    map_df["price_eur"] = pd.to_numeric(map_df["price_eur"], errors="coerce")

    # Taille des points ~ prix
    r_min, r_max = 25, 150
    pmin, pmax = map_df["price_eur"].min(), map_df["price_eur"].max()
    map_df["radius"] = (r_min + r_max)/2 if pmin == pmax else r_min + (map_df["price_eur"]-pmin)*(r_max-r_min)/(pmax-pmin)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius="radius",
        get_color=[255, 99, 71],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(map_df["lat"].mean()),
        longitude=float(map_df["lon"].mean()),
        zoom=11, pitch=0, bearing=0
    )

    # Fond CARTO (pas de token nécessaire) + infobulle claire
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip={
            "html": (
                "<b>Adresse :</b> {address}<br/>"
                "<b>Prix :</b> {price_eur} €<br/>"
                "<a href='{url}' target='_blank'>Annonce</a>"
            ),
            "style": {
                "backgroundColor": "#f9f9f9",
                "color": "#333333",
                "fontSize": "13px",
                "border": "1px solid #cccccc",
                "borderRadius": "6px",
                "padding": "6px 8px",
                "boxShadow": "0px 2px 6px rgba(0,0,0,0.15)"
            },
        },
    )

    st.markdown("### 🗺️ Carte")
    st.pydeck_chart(deck, use_container_width=True)
    return True


# On force : pas de géocodage automatique
do_geocode = False

# A) 1er essai d'affichage avec les coordonnées déjà présentes
shown = _try_show_map(dff)

# B) Si rien à afficher, on complète d'abord par le code postal Paris, puis on réessaie
if not shown and "postal_code" in dff.columns:
    missing = dff["lat"].isna() | dff["lon"].isna()
    for idx, cp in dff.loc[missing, "postal_code"].astype(str).items():
        cp = cp.strip()
        if cp in PARIS_ARR_COORDS:
            y, x = PARIS_ARR_COORDS[cp]
            dff.at[idx, "lat"] = y
            dff.at[idx, "lon"] = x
    shown = _try_show_map(dff)

# C) Optionnel : géocoder → jamais exécuté (do_geocode = False)
if not shown and do_geocode and "address" in dff:
    st.caption("Géocodage en cours (Nominatim)…")
    lat_new, lon_new = [], []
    for addr in dff["address"].fillna(""):
        y, x = geocode_address(addr)
        lat_new.append(y); lon_new.append(x)
    dff.loc[dff["lat"].isna(), "lat"] = pd.Series(lat_new, index=dff.index)[dff["lat"].isna()]
    dff.loc[dff["lon"].isna(), "lon"] = pd.Series(lon_new, index=dff.index)[dff["lon"].isna()]
    shown = _try_show_map(dff)

# ------------------ FIN CARTE ------------------

# Tableau
st.markdown("### 📋 Données filtrées")
# Configuration des colonnes (URL cliquable si possible)
col_config = {}
if "url" in dff.columns:
    try:
        col_config["url"] = st.column_config.LinkColumn("Lien", display_text="Annonce")
    except Exception:
        pass

st.dataframe(
    dff.sort_values(by=["price_eur"], ascending=True, na_position="last") if "price_eur" in dff else dff,
    use_container_width=True,
    column_config=col_config or None,
)

st.caption("Astuce : mets à jour l’URL du CSV dans la barre latérale pour pointer sur ta dernière donnée.")
