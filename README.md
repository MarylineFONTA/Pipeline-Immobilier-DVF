
# 🏠 Pipeline Immobilier – DVF

Projet réalisé dans le cadre de la formation **PMN 2025**.

Pipeline complet de **collecte → nettoyage → visualisation** des données immobilières (type **DVF**) avec un **tableau de bord Streamlit**.

* **`src/spider.py`** : collecte (scraping / téléchargement) → produit un **JSON brut**
* **`src/cleaner.py`** : nettoyage & enrichissement → produit un **CSV propre**
* **`src/app.py`** : application **Streamlit** (KPI, carte, filtres)
* **GitHub Actions (CI/CD)** : automatisation du pipeline et publication des données dans `data/`

---

## 📦 Arborescence

```
PIPELINE-IMMOBILIER - DVF/
├─ .github/
│  └─ workflows/
│     └─ main.yml                 # pipeline CI/CD (automatisation)
├─ data/
│  ├─ cleaned_data.csv            # données nettoyées (source du dashboard)
│  ├─ raw_data.json               # données brutes
│  └─ __init__.py
├─ src/
│  ├─ app.py                      # application Streamlit
│  ├─ cleaner.py                  # script de nettoyage
│  └─ spider.py                   # script d’acquisition
├─ requirements.txt               # dépendances Python
└─ README.md
```

---

## ✨ Fonctionnalités principales

### Acquisition

* Génération d’un fichier brut `raw_data.json` (DVF ou autre source).

### Nettoyage

* Harmonisation des colonnes : `price_eur`, `surface_m2`, `lat`, `lon`, `address`, `postal_code`, `city`
* Calcul automatique `price_per_m2`
* Correction des erreurs fréquentes :

  * micro-degrés (`/1e6` ou `/1e5`)
  * inversion lat/lon (détection heuristique France)

### Visualisation (Streamlit)

* KPIs globaux et filtrés
* Histogramme des prix
* Carte **PyDeck** centrée sur l’Île-de-France (points fixes en pixels)
* Filtres : prix, surface, ville (Arrondissement), recherche par adresse
* Bouton **↻ Recharger les données** (vide cache local + contourne le cache GitHub CDN)

### Automatisation (CI/CD)

* Pipeline exécuté automatiquement via **GitHub Actions**
* Mise à jour régulière des données (`cleaned_data.csv`)
* Déploiement continu sur **Streamlit Cloud**

---

## 🛠️ Installation locale

> Python **3.10+** recommandé.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### 1) Lancer le dashboard

```bash
streamlit run src/app.py
```

L’app charge par défaut `data/cleaned_data.csv` (ou une URL GitHub *raw* si configurée dans l’UI).
Tu peux coller une URL *raw* GitHub dans la barre latérale pour tester un fichier à jour.

### 2) Exécuter le pipeline complet manuellement

```bash
# Acquisition
python src/spider.py --city "Paris" --year-min 2019 --year-max 2024 -o data/raw_data.json

# Nettoyage / normalisation
python src/cleaner.py -i data/raw_data.json -o data/cleaned_data.csv
```

Relance ensuite l’app ou clique **↻ Recharger les données**.

---

## 🤖 Automatisation (CI/CD avec GitHub Actions)

Le pipeline peut tourner automatiquement grâce à GitHub Actions :

* Exécution déclenchée par un push
* Mise à jour de `data/cleaned_data.csv`
* Déploiement automatique sur Streamlit Cloud


## 🌐 Déploiement

App Streamlit hébergée ici :
👉 [pipeline-immobilier-dvf-ernest-maryline.streamlit.app](https://pipeline-immobilier-dvf-ernest-maryline.streamlit.app/)

---

## 🧭 Feuille de route (idées)

* Export CSV/XLSX des filtres courants
* Sélecteur d’année/période
* Clustering des points sur la carte
* Comparaison arrondissement / ville


---

## 📜 Licence & crédits

* Données DVF (Etalab) — usage conforme aux licences open data.
* Code : licence libre.


