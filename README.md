# PipeLine-Immobilier
Projet lors de la formation PMN  2025


---

# Pipeline Immobilier – DVF

Pipeline de collecte → nettoyage → visualisation des données immobilières (type **DVF**) avec un tableau de bord **Streamlit**.

* **`src/spider.py`** : collecte (scraping / téléchargement) → produit un **JSON brut**
* **`src/cleaner.py`** : normalisation & enrichissement → produit un **CSV propre**
* **`src/app.py`** : application **Streamlit** pour explorer les données (KPI, carte, filtres)
* **GitHub Actions** : automatisation possible du pipeline & des données publiées dans `data/`

---

## 📦 Arborescence

```
PIPELINE-IMMOBILIER - DVF/
├─ .github/
│  └─ workflows/
│     └─ main.yml                 # pipeline CI/CD (optionnel)
├─ data/
│  ├─ cleaned_data.csv            # données nettoyées (source du dashboard)
│  ├─ raw_data.json               # données brutes
│  └─ __init__.py
├─ src/
│  ├─ app.py                      # application Streamlit
│  ├─ cleaner.py                  # script de nettoyage / normalisation
│  └─ spider.py                   # script d’acquisition (scraper / fetch)
├─ requirements.txt               # dépendances Python
└─ README.md
```

---

## ✨ Fonctionnalités principales

* **Acquisition** : construction d’un fichier brut `raw_data.json` (DVF ou autre source)
* **Nettoyage** :

  * Harmonisation des colonnes : `price_eur`, `surface_m2`, `lat`, `lon`, `address`, `postal_code`, `city`
  * Calcul `price_per_m2` quand pertinent
  * Correction automatique des coordonnées fréquentes :

    * micro-degrés → **division par 1e6 / 1e5**
    * inversion **lat/lon** (détection heuristique France)
* **Visualisation** (Streamlit) :

  * KPIs globaux & filtrés
  * **Histogramme** des prix
  * **Carte PyDeck** centrée sur **Île-de-France** (filtrage bbox IdF), points en **pixels** (ne grossissent pas au zoom)
  * **Tooltip** : Ville, Adresse, Prix
  * **Filtres** :

    * Prix (€) : **slider** 
    * Surface (m²), Ville (multiselect), recherche dans l’adresse
  * Bouton **↻ Recharger les données** (vide cache Streamlit **et** contourne le cache CDN GitHub)
* **Robustesse GitHub** :

  * Conversion automatique `github.com/.../blob/...` → `raw.githubusercontent.com/...`
  * Récupération de la **date du dernier commit** (pour afficher “données mises à jour le …” et casser les caches)

---

## 🛠️ Installation

> Python **3.10+** recommandé.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Démarrage rapide

### 1) Lancer le dashboard (avec le CSV fourni)

```bash
streamlit run src/app.py
```

Par défaut, l’app charge `data/cleaned_data.csv` (ou une URL GitHub *raw* si configurée dans l’UI).
Dans la barre latérale, tu peux coller une autre URL *raw* GitHub pour tester un fichier à jour.

### 2) Exécuter le pipeline complet

> Les scripts `spider.py` et `cleaner.py` peuvent proposer des options.
> Consulte **l’aide intégrée** :

```bash
python src/spider.py -h
python src/cleaner.py -h
```

Exemple de flux (à adapter à tes options réelles) :

```bash
# 1) Acquisition (écrit data/raw_data.json)
python src/spider.py --city "Paris" --year-min 2019 --year-max 2024 -o data/raw_data.json

# 2) Nettoyage / normalisation (écrit data/cleaned_data.csv)
python src/cleaner.py -i data/raw_data.json -o data/cleaned_data.csv
```

Ensuite, relance l’app Streamlit ou clique **↻ Recharger les données**.


## 🚀 Démarrage du dashboard automatisé

Lien streamlit : https://pipeline-immobilier-dvf-ernest-maryline.streamlit.app/

---
## ⚙️ Configuration & variables

### Secrets (facultatif mais recommandé)

* `GITHUB_TOKEN` : augmente la fiabilité/rapidité pour récupérer la date du dernier commit via l’API GitHub
  (utile si tu déploies sur Streamlit Cloud ou si tu fais beaucoup d’actualisations).

### Paramètres carte (dans `src/app.py`)

* Centre & emprise **Île-de-France** :

  ```python
  IDF_CENTER_LAT, IDF_CENTER_LON = 48.8566, 2.3522
  IDF_BBOX = dict(lat_min=48.0, lat_max=49.2, lon_min=1.3, lon_max=3.6)
  ```

  Modifie ces bornes si tu souhaites élargir/réduire l’affichage.

### Colonnes attendues (après nettoyage)

| Colonne        | Description                                 |
| -------------- | ------------------------------------------- |
| `price_eur`    | Prix en euros (int/float)                   |
| `surface_m2`   | Surface en m²                               |
| `lat`, `lon`   | Coordonnées géographiques en **degrés**     |
| `address`      | Adresse lisible (tooltip)                   |
| `postal_code`  | Code postal (5 chiffres)                    |
| `city`         | Ville affichée au format **`Ville (CP)`**   |
| `price_per_m2` | Calculé quand `price_eur` & `surface_m2` OK |

> Si `city` est manquante, elle est reconstruite depuis `address` :
> ex. “Paris 16e Arrondissement (75116)” → **“Paris (75116)”**.

---

## 🤖 Intégration continue (GitHub Actions)

Le fichier `.github/workflows/main.yml` peut :

* Installer les dépendances
* Lancer `spider.py` puis `cleaner.py`
* Commiter/pousser le nouveau `data/cleaned_data.csv` (selon ta config)

> Ouvre `main.yml` et adapte les jobs (triggers `on:` push/schedule, secrets, etc.).
> **Astuce** : pour commiter depuis l’action, configure `permissions: contents: write` et utilise `actions/checkout` + `git` CLI.

---

## 🗺️ Points importants sur la carte

* La couche utilise `radius_units="pixels"` → taille **constante** à l’écran (ne grossit pas au zoom).
* Tu peux ajouter un slider “Taille des points (px)” dans la sidebar si besoin.
* Les points hors IdF sont **ignorés** à l’affichage (filtre bbox).
  Si certains biens légitimes disparaissent, élargis `IDF_BBOX`.

---

## 🧼 Stratégies de qualité des données

* **Micro-degrés** détectés automatiquement → division `1e6` (ou `1e5`)
* **Lat/Lon inversés** détectés (médiane lat≈2.x & lon≈48.x en France) → swap
* **Dédoublonnage optionnel** par coordonnées exactes :
  décommente dans `normalize_columns()` si tu veux éviter les points superposés :

  ```python
  # df = df.drop_duplicates(subset=["lat", "lon"], keep="first")
  ```

---

## 🧯 Dépannage

* **Je vois encore d’anciennes colonnes/valeurs**
  → clique **↻ Recharger les données** (vide `st.cache_data` **et** contourne le CDN GitHub).
  En dev : `streamlit cache clear`.

* **La carte n’apparaît pas**
  → vérifie qu’il existe des lignes avec `lat` **et** `lon` valides **et** dans la bbox IdF.
  Ajoute temporairement dans l’app :

  ```python
  st.caption(f"lat [{dff['lat'].min():.5f} .. {dff['lat'].max():.5f}] | "
             f"lon [{dff['lon'].min():.5f} .. {dff['lon'].max():.5f}]")
  ```

* **`{address}` ne s’affiche pas dans le tooltip**
  → assure-toi que la colonne `address` est bien incluse dans le `map_df` passé à la couche PyDeck.


---

## 🧪 Tests manuels rapides

1. Ouvre l’app → vérifie que les KPI & l’histogramme s’affichent.
2. Recherche “Boulevard”, “Gambetta”… → la liste/ carte doit se filtrer.
3. Clique **↻ Recharger les données** après avoir mis à jour `cleaned_data.csv` sur GitHub → les KPI doivent évoluer.

---

## 📜 Licence & crédits

* Données DVF (Etalab) — respecte les termes d’usage des données publiques.
* Code : choisis la licence adaptée (MIT conseillé si open-source).

---

## 🧭 Feuille de route (idées)

* Export CSV/XLSX des filtres courants
* Selecteur **d’année** / période
* Clustering des points sur la carte
* Comparaison arrondissement / ville
* Déploiement Streamlit Cloud / Docker

---

