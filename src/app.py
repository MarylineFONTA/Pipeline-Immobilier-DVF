# src/app.py
from __future__ import annotations

import csv
import io
import os
import re
from datetime import datetime
from functools import lru_cache

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# ================== CONFIG ==================
st.set_page_config(page_title="Immo Dashboard (DVF)", layout="wide")

DEFAULT_CSV_URL = (
    "https://raw.githubusercontent.com/MarylineFONTA/Pipeline-Immobilier-DVF/"
    "refs/heads/main/data/cleaned_data.csv"
)

CACHE_TTL = 600  # 10 min


# ================== UTILS ==================

# ---- Carte : centre + bbox Île-de-France ----
IDF_CENTER_LAT, IDF_CENTER_LON = 48.8566, 2.3522
IDF_BBOX = dict(lat_min=48.0, lat_max=49.2, lon_min=1.3, lon_max=3.6)

@st.cache_data(show_spinner=True, ttl=CACHE_TTL)
def get_csv_last_modified(url: str) -> datetime | None:
    """Date locale du dernier commit ayant modifié le fichier (GitHub)."""
    def to_local(dt: datetime) -> datetime:
        try:
            from zoneinfo import ZoneInfo
            return dt.astimezone(ZoneInfo("Europe/Paris"))
        except Exception:
            return dt.astimezone()

    def parse_github(u: str):
        if "raw.githubusercontent.com" in u:
            parts = u.split("raw.githubusercontent.com/")[-1].split("/")
            owner, repo = parts[0], parts[1]
            if len(parts) >= 5 and parts[2] == "refs" and parts[3] == "heads":
                branch = parts[4]; path = "/".join(parts[5:])
            else:
                branch = parts[2]; path = "/".join(parts[3:])
            return owner, repo, branch, path
        if "github.com" in u and "/blob/" in u:
            tail = u.split("github.com/")[-1]
            owner, repo, _, branch, *pp = tail.split("/")
            return owner, repo, branch, "/".join(pp)
        return None

    parsed = parse_github(url)
    if not parsed:
        # Fallback générique
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
    api = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"path": path, "sha": branch, "per_page": 1}
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "immo-dashboard"}

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        try:
            token = st.secrets.get("GITHUB_TOKEN", None)
        except Exception:
            token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.get(api, params=params, headers=headers, timeout=15)
        if r.ok and r.json():
            iso = r.json()[0]["commit"]["committer"]["date"]
            return to_local(datetime.fromisoformat(iso.replace("Z", "+00:00")))
    except Exception:
        pass
    return None


def _cache_bust(url: str, version: int | None) -> str:
    if version is None:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}v={version}"


@st.cache_data(show_spinner=True, ttl=CACHE_TTL)
def load_csv(url: str, version: int | None = None) -> pd.DataFrame:
    """
    Charge un CSV GitHub (blob/raw). Ignore une 1re ligne 'sep=' éventuelle.
    Normalise les colonnes.
    """
    # Convertit "blob" -> "raw"
    if url.startswith("http") and "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

    # Casse le cache CDN si on a une version
    url = _cache_bust(url, version)

    headers = {"User-Agent": "immo-dashboard", "Cache-Control": "no-cache", "Pragma": "no-cache"}
    if url.startswith("http"):
        text = requests.get(url, timeout=30, headers=headers).text
    else:
        with open(url, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("sep="):
        text = "\n".join(lines[1:])

    buf = io.StringIO(text)
    df = pd.read_csv(
        buf,
        sep=";",
        engine="python",
        encoding="utf-8",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="error",
    )
    return normalize_columns(df)


def normalize_city(addr: str) -> str | None:
    if not addr or not isinstance(addr, str):
        return None

    # 1) récupérer le code postal
    m_cp = re.search(r"\b(\d{5})\b", addr)
    cp = m_cp.group(1) if m_cp else ""

    # 2) récupérer la ville
    # Cas spécial Paris
    if "Paris" in addr:
        return f"Paris ({cp})" if cp else "Paris"

    # Ville générique : prend le mot avant le code postal ou avant la parenthèse
    m_city = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ'’\- ]+)\s*(?:\(|\d{5})", addr)
    city = m_city.group(1).strip() if m_city else None

    if city and cp:
        return f"{city} ({cp})"
    return city

def normalize_city_from_address(addr: str) -> str | None:
    """Extrait 'Ville (CP)' depuis l'adresse."""
    if not addr or not isinstance(addr, str):
        return None
    m_cp = re.search(r"\b(\d{5})\b", addr)
    cp = m_cp.group(1) if m_cp else None
    # capture la ville avant la parenthèse ou avant le CP
    m_city = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ'’\- ]+)\s*(?:\(|\d{5})", addr)
    city = m_city.group(1).strip() if m_city else None
    return normalize_city(city, cp)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise les colonnes, corrige l'échelle lat/lon, détecte lat/lon inversés,
    et calcule price_per_m2 si possible."""
    cols_low = {c.lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    def has(*names): return next((cols_low[n] for n in names if n in cols_low), None)

    c_price = has("prix_eur", "price_eur", "prix")
    c_surf  = has("surface_m2", "surface", "surface_m2 (m2)")
    c_addr  = has("address", "adresse", "location")
    c_cp    = has("postal_code", "cp", "code_postal")
    c_lat   = has("lat", "latitude", "y")
    c_lon   = has("lon", "lng", "longitude", "x")
    c_url   = has("url", "lien")
    c_city  = has("city", "ville")

    if c_price: rename[c_price] = "price_eur"
    if c_surf:  rename[c_surf]  = "surface_m2"
    if c_addr:  rename[c_addr]  = "address"
    if c_cp:    rename[c_cp]    = "postal_code"
    if c_lat:   rename[c_lat]   = "lat"
    if c_lon:   rename[c_lon]   = "lon"
    if c_url:   rename[c_url]   = "url"
    if c_city:  rename[c_city]  = "city"

    df = df.rename(columns=rename)

    # numériques
    for c in ("price_eur", "surface_m2", "lat", "lon"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Corriger l'échelle lat/lon (micro-degrés -> degrés) ---
    if {"lat","lon"}.issubset(df.columns):
        lat_med = df["lat"].abs().median(skipna=True)
        lon_med = df["lon"].abs().median(skipna=True)
        if pd.notna(lat_med) and pd.notna(lon_med) and (lat_med > 90 or lon_med > 180):
            maxv = float(max(lat_med, lon_med))
            factor = 1e6 if maxv >= 1e6 else (1e5 if maxv >= 1e5 else 1.0)
            if factor != 1.0:
                df["lat"] = df["lat"] / factor
                df["lon"] = df["lon"] / factor

        # --- Détecter lat/lon inversés (typique : lat≈2.x, lon≈48.x en France) ---
        lat_med = df["lat"].median(skipna=True)
        lon_med = df["lon"].median(skipna=True)
        if pd.notna(lat_med) and pd.notna(lon_med):
            if (-5 <= lat_med <= 10) and (41 <= lon_med <= 51):
                df[["lat", "lon"]] = df[["lon", "lat"]]

    # €/m² si possible
    if {"price_eur", "surface_m2"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["price_per_m2"] = (df["price_eur"] / df["surface_m2"]).replace([np.inf, -np.inf], np.nan)
        df["price_per_m2"] = df["price_per_m2"].round(0)

    # City -> "Ville (CP)" depuis l'adresse
    if "address" in df:
        df["city"] = df["address"].fillna("").apply(normalize_city)

    return df




def _safe_range(min_v, max_v, default_span=10, floor=0):
    """Retourne (lo, hi) sûr même si NaN, inversé ou égal."""
    try:
        mn = int(min_v) if pd.notna(min_v) else None
        mx = int(max_v) if pd.notna(max_v) else None
    except Exception:
        mn, mx = None, None
    if mn is None or mx is None:
        return floor, floor + default_span
    if mn > mx:
        mn, mx = mx, mn
    if mn == mx:
        lo = max(floor, mn - default_span // 2)
        hi = lo + max(1, default_span)
        return lo, hi
    return mn, mx


# ================== SIDEBAR ==================
st.sidebar.header("Paramètres")
csv_url = st.sidebar.text_input(
    "URL CSV (GitHub raw)",
    value=DEFAULT_CSV_URL,
    help="Ex : https://github.com/…/blob/main/data/cleaned_data.csv",
)

if st.sidebar.button("↻ Recharger les données"):
    # vider les caches + forcer une version unique
    get_csv_last_modified.clear()
    load_csv.clear()
    st.cache_data.clear()
    import time
    st.session_state["force_version"] = int(time.time())
    st.rerun()

# version basée sur le dernier commit (ou forçage bouton)
last_dt = get_csv_last_modified(csv_url)
version_from_commit = int(last_dt.timestamp()) if last_dt else None
force_version = st.session_state.get("force_version")
version = force_version or version_from_commit

df = load_csv(csv_url, version=version)

st.sidebar.markdown("### Filtres")
raw_price_min = df["price_eur"].min() if "price_eur" in df else np.nan
raw_price_max = df["price_eur"].max() if "price_eur" in df else np.nan
price_eur_min, price_eur_max = _safe_range(raw_price_min, raw_price_max, default_span=10000, floor=0)

raw_surf_min = df["surface_m2"].min() if "surface_m2" in df else np.nan
raw_surf_max = df["surface_m2"].max() if "surface_m2" in df else np.nan
surface_m2_min, surface_m2_max = _safe_range(raw_surf_min, raw_surf_max, default_span=20, floor=0)

def fmt_fr(n): return f"{int(n):,}".replace(",", " ")

step = int(max(1000, (price_eur_max - price_eur_min)//100 or 1))
options = list(range(price_eur_min, price_eur_max + 1, step))
if options[-1] != price_eur_max:
    options.append(price_eur_max)
if len(options) < 2:
    options = [price_eur_min, price_eur_min + 1]

'''price_eur_sel = st.sidebar.select_slider(
    "Prix (€)",
    options=options,
    value=(options[0], options[-1]),
    format_func=lambda x: f"{fmt_fr(x)} €",
    key=f"price_{options[0]}_{options[-1]}",
)'''
# ==== Prix (€) : number inputs + slider synchronisés ====
st.sidebar.markdown("#### Prix (€)")

# Série des prix nettoyée
s = pd.to_numeric(df.get("price_eur"), errors="coerce").dropna().astype(int)
if len(s) == 0:
    price_options = [0, 1]
else:
    qs = np.linspace(0, 1, 51)  # ~50 crans
    price_options = sorted(set(int(np.quantile(s, q)) for q in qs))
    # garantir que les bornes min/max réelles sont incluses
    price_options[0] = int(s.min())
    price_options[-1] = int(s.max())

price_min, price_max = st.sidebar.select_slider(
    "Prix (€)",
    options=price_options,
    value=(price_options[0], price_options[-1]),
    format_func=lambda x: f"{fmt_fr(x)} €",
    key="price_select_quantiles",
)
surface_m2_sel = st.sidebar.slider(
    "Surface (m²)",
    min_value=surface_m2_min,
    max_value=surface_m2_max,
    value=(surface_m2_min, surface_m2_max),
    step=1,
)

cities = sorted([c for c in df.get("city", pd.Series([])).dropna().unique().tolist()])
city_sel = st.sidebar.multiselect("Ville", cities, default=[])
q = st.sidebar.text_input("Recherche texte (dans l’adresse)", value="")

# ================== MAIN ==================
st.markdown("<style>.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

last_txt = last_dt.strftime("%d/%m/%Y %H:%M") if last_dt else "indisponible"
st.markdown(
    f"""
    <h1 style='text-align:left; font-size:38px; color:#2C3E50;'>🏠 Tableau de bord immobilier - Paris</h1>
    <p style='text-align:left; font-size:16px; color:gray; margin-top:-10px;'>
        (Source : DVF&nbsp;•&nbsp;🗓️ Données mises à jour : {last_txt})
    </p>
    """,
    unsafe_allow_html=True,
)

# KPIs globaux
left, mid, right = st.columns(3)
left.metric("Annonces (total)", len(df))
if "price_eur" in df:
    mid.metric("Prix moyen (global)", f"{int(df['price_eur'].mean(skipna=True)):,} €".replace(",", " "))
if "price_per_m2" in df:
    right.metric("€/m² moyen (global)", f"{int(df['price_per_m2'].mean(skipna=True)):,} €".replace(",", " "))

# Filtres
mask = pd.Series(True, index=df.index)
if "price_eur" in df:
    mask &= df["price_eur"].between(price_min, price_max, inclusive="both")
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
if "price_per_m2" in dff and len(dff):
    k3.metric("€/m² moyen (filtré)", f"{int(dff['price_per_m2'].mean(skipna=True)):,} €".replace(",", " "))

# Histogramme prix (5 classes)
if "price_eur" in dff and dff["price_eur"].notna().any():
    st.markdown("### 📈 Histogramme des prix (5 classes)")
    s = pd.to_numeric(dff["price_eur"], errors="coerce").dropna()
    if len(s):
        lo, hi = float(s.min()), float(s.max())
        edges = np.array([lo - 0.5, hi + 0.5]) if np.isclose(lo, hi) else np.linspace(lo, hi, 6)
        cats = pd.cut(s, bins=edges, include_lowest=True, right=True)
        counts = cats.value_counts(sort=False)
        labels = [f"{int(edges[i]):,} – {int(edges[i+1]):,} €".replace(",", " ") for i in range(len(edges)-1)]
        chart_df = pd.DataFrame({"Intervalle": pd.Categorical(labels, categories=labels, ordered=True),
                                 "Nombre": counts.values})
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Intervalle:N", sort=labels, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("Nombre:Q", axis=alt.Axis(title=None)),
            tooltip=["Intervalle", "Nombre"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Pas de prix exploitables pour l’histogramme.")

# -------------------- CARTE --------------------
# ⚠️ On n'utilise QUE les lat/lon du fichier. Aucune substitution/estimation.
def _try_show_map(df_: pd.DataFrame) -> bool:
    if not {"lat", "lon"}.issubset(df_.columns):
        return False

    valid = (
        df_["lat"].between(-90, 90)
        & df_["lon"].between(-180, 180)
        & df_[["lat", "lon"]].notna().all(axis=1)
    )
    if not valid.any():
        return False

    # 👉 Inclure l'adresse si elle existe
    cols = ["lat", "lon", "price_eur", "city"]
    if "address" in df_.columns:
        cols.append("address")

    map_df = df_.loc[valid, cols].copy()

    # Texte propre pour le tooltip
    if "address" in map_df.columns:
        map_df["address"] = map_df["address"].fillna("—").astype(str)

    map_df["radius_px"] = np.interp(
        map_df["price_eur"].fillna(map_df["price_eur"].median()),
        (map_df["price_eur"].min(), map_df["price_eur"].max()),
        (3, 12)  # min / max en pixels
    )


    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius="radius_px",       # <- colonne calculée ci-dessus
        radius_units="pixels",
        stroked=True,
        get_line_color=[255, 99, 71],
        line_width_min_pixels=1,
        filled=True,
        get_fill_color=[255, 99, 71, 140],
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=10),
        map_provider="carto",
        map_style="light",
        tooltip={"html": "<b>Ville :</b> {city}<br/><b>Adresse :</b> {address}<br/><b>Prix :</b> {price_eur} €"},
    )
    st.pydeck_chart(deck, use_container_width=True)
    return True


shown = _try_show_map(dff)
if not shown:
    st.warning("Aucun point avec coordonnées valides (lat & lon) dans les données filtrées.")

# ================== TABLEAU ==================
st.markdown("### 📋 Données filtrées")
st.dataframe(
    dff.sort_values(by=["price_eur"], ascending=True, na_position="last") if "price_eur" in dff else dff,
    use_container_width=True,
    column_config={
        "city": st.column_config.TextColumn("Ville", width="large"),
    },
)
st.caption("Astuce : mets à jour l’URL du CSV dans la barre latérale pour pointer sur ta dernière donnée.")
