# src/cleaner.py
#from __future__ import annotations

import json
import re
import csv
from pathlib import Path
from typing import List, Dict
import pandas as pd


def read_json_records(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8").strip()
     # Si le fichier est vide, renvoyer une liste vide pour éviter l'erreur
    if not text:
        print(f"ℹ Fichier JSON vide trouvé, renvoi d'une liste vide : {path}")
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("Le JSON n'est pas une liste d'objets.")
    except json.JSONDecodeError:
        # Fallback NDJSON
        records: List[Dict] = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError
                records.append(obj)
            except Exception as exc:
                raise ValueError(f"Ligne {i}: impossible de parser en JSON.") from exc
        return records


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = ["price_eur", "surface_m2","lon", "lat"]
    int_cols = ["rooms", "floor", "year_built"]
    str_cols = ["source", "title", "address", "postal_code","city", "property_type"]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def add_price_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    if "price_eur" not in df.columns or "surface_m2" not in df.columns:
        df["price_per_m2"] = pd.Series([pd.NA] * len(df), dtype="Float64")
        return df

    price = pd.to_numeric(df["price_eur"], errors="coerce")
    surface = pd.to_numeric(df["surface_m2"], errors="coerce")
    mask = price.notna() & surface.notna() & (surface > 0)

    ppm2 = pd.Series([pd.NA] * len(df), dtype="Float64")
    ppm2[mask] = (price[mask] / surface[mask]).round(2)
    df["price_per_m2"] = ppm2
    return df


def sanitize_strings(df: pd.DataFrame, sep: str = ";") -> pd.DataFrame:
    """
    Supprime retours ligne/tabulations, compresse les espaces et remplace le séparateur
    éventuel dans les colonnes texte pour garantir 1 ligne physique par annonce.
    """
    text_cols = df.select_dtypes(include=["string", "object"]).columns

    def clean(x):
        if pd.isna(x):
            return x
        s = str(x)
        # Enlever \r, \n, \t et compresser espaces
        s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # Éviter que le séparateur apparaisse dans le texte
        if sep in s:
            s = s.replace(sep, ",")
        return s

    for c in text_cols:
        df[c] = df[c].map(clean)
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[1]   # dossier racine du projet
    data_dir = root / "data"
    json_in = data_dir / "raw_data.json"
    csv_out = data_dir / "cleaned_data.csv"

    if not json_in.exists():
        raise SystemExit(f"Fichier introuvable : {json_in}")

    records = read_json_records(json_in)
    if not records:
        raise SystemExit("Aucune annonce trouvée dans le JSON.")

    df = pd.DataFrame(records)
    df = coerce_types(df)
    df = add_price_per_m2(df)

    # Nettoyage des colonnes texte
    df = sanitize_strings(df, sep=";")

    # 🔽 Ici on enlève les doublons stricts sur toutes les colonnes
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"✔ Déduplication : {before} → {after} lignes")

    # Colonnes ordonnées
    ordered_cols = [
        "source","ID", "date","title", "postal_code", "address","city",
        "rooms", "floor", "surface_m2",
        "price_eur", "price_per_m2",
        "property_type",
        "lon","lat"
    ]
    existing = [c for c in ordered_cols if c in df.columns]
    df = df[existing]

    # Stat globale (optionnel)
    avg_ppm2 = df["price_per_m2"].mean(skipna=True)
    avg_msg = "indisponible"
    if pd.notna(avg_ppm2):
        avg_msg = f"{avg_ppm2:,.2f} €/m²".replace(",", " ").replace(".", ",")

    # Écriture CSV
    df.to_csv(
        csv_out,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    print(f"✔ CSV écrit : {csv_out}")
    print(f"✔ Lignes (annonces) : {len(df)}")
    print(f"ℹ Prix moyen au m² : {avg_msg}")


if __name__ == "__main__":
    main()
