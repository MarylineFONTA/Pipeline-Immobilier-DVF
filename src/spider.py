# src/spider.py
# Spider DVF (Open Data Etalab) — télécharge les fichiers CSV.gz par département/année
# et écrit un fichier de sortie (JSON par défaut) via Scrapy FEEDS.

import csv
import gzip
import io
from urllib.parse import urljoin
from pathlib import Path
import argparse

import scrapy
from scrapy.crawler import CrawlerProcess

# Base des fichiers DVF transformés (Geo-DVF)
# Arbo typique: .../latest/csv/<ANNEE>/departements/<DEP>.csv.gz
DVF_BASE = "https://files.data.gouv.fr/geo-dvf/latest/csv/"

def to_float_fr(x):
    if x is None:
        return None
    s = str(x).replace("\u202f","").replace("\u00A0","").replace(" ","")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


class DVFSpider(scrapy.Spider):
    """
    Exemples :
      
      # scrapy crawl dvf -a departement=75 -a years=2024 -O data/raw_data.json

    """
    name = "dvf"
    allowed_domains = ["files.data.gouv.fr", "data.gouv.fr", "gouv.fr"]
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.2,
        "CONCURRENT_REQUESTS": 2,
        "FEED_EXPORT_ENCODING": "utf-8",
        "USER_AGENT": "Mozilla/5.0 (compatible; dvf-scraper/1.0)",
    }

    def __init__(self, departement="75", years="2024", limit=None, *args, **kwargs):
        """
        departement : code département (ex. 75, 92, 13…)
        years       : liste CSV d'années (ex. "2022,2023")
        limit       : optionnel, nombre max de lignes (tests)
        """
        super().__init__(*args, **kwargs)
        self.dept = str(departement).zfill(2)
        self.years = [y.strip() for y in str(years).split(",") if y.strip()]
        self.limit = int(limit) if limit else None
        self._count = 0

    def start_requests(self):
        for y in self.years:
            url = urljoin(DVF_BASE, f"{y}/departements/{self.dept}.csv.gz")
            yield scrapy.Request(url, callback=self.parse_dvf_file, meta={"year": y}, dont_filter=True)

    def parse_dvf_file(self, response):
        year = response.meta.get("year")
        if response.status != 200:
            self.logger.warning(f"DVF {year}/{self.dept} introuvable ({response.status}) -> {response.url}")
            return

        # Décompresse si .gz, sinon lit tel quel
        body = response.body
        if response.url.endswith(".gz"):
            try:
                body = gzip.decompress(body)
            except OSError:
                self.logger.warning("Décompression gzip impossible, tentative lecture brute.")

        # Stream texte
        text_io = io.TextIOWrapper(io.BytesIO(body), encoding="utf-8", errors="replace")

        # Détection du séparateur (',' ou ';')
        sample = text_io.read(8192)
        text_io.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        except csv.Error:
            class _D: delimiter = ";"
            dialect = _D()

        reader = csv.DictReader(text_io, dialect=dialect)

        # Champs usuels DVF: valeur_fonciere, surface_reelle_bati, nombre_pieces_principales,
        # date_mutation, type_local, code_postal, nom_commune,
        # adresse_numero, adresse_nom_voie, longitude, latitude, id_mutation, numero_disposition, code_departement
        for row in reader:
            if self.limit and self._count >= self.limit:
                break

            try:
                # Filtre de sécurité sur le département si présent
                if row.get("code_departement") and str(row["code_departement"]).zfill(2) != self.dept:
                    continue

                numero  = (row.get("adresse_numero") or "").strip()
                voie    = (row.get("adresse_nom_voie") or "").strip()
                cp      = (row.get("code_postal") or "").strip()
                commune = (row.get("nom_commune") or row.get("commune") or "").strip()

                part_voie = f"{numero} {voie}".strip()
                part_comm = f"{commune} ({cp})" if cp else commune
                address   = ", ".join([p for p in [part_voie, part_comm] if p])

                # ID stable DVF
                mut_id = (row.get("id_mutation") or "").strip()
                dispo  = (row.get("numero_disposition") or "").strip()
                uid = f"{mut_id}-{dispo}" if mut_id and dispo else (mut_id or None)

                item = {
                    "source"        : "dvf",
                    "ID"            : uid,
                    "date"          : row.get("date_mutation"),
                    "price_eur"     : to_float_fr(row.get("valeur_fonciere")),
                    "surface_m2"    : to_float_fr(row.get("surface_reelle_bati")),
                    "rooms"         : int(row["nombre_pieces_principales"]) if row.get("nombre_pieces_principales") else None,
                    "property_type" : (row.get("type_local") or "").strip().lower(),
                    "address"       : address or None,
                    "postal_code"   : cp or None,
                    "city"          : commune or None,
                    "lon"           : to_float_fr(row.get("longitude")),
                    "lat"           : to_float_fr(row.get("latitude")),
                }

                # On ne garde que les mutations bâties avec prix + surface
                if item["price_eur"] is None or item["surface_m2"] is None:
                    continue

                self._count += 1
                yield item

            except Exception as e:
                self.logger.debug(f"Ligne ignorée ({e})")
                continue

''
# --- Exécution directe ----------------
def _run_standalone():
    parser = argparse.ArgumentParser(description="DVF spider (standalone)")
    parser.add_argument("--departement", default="75")
    parser.add_argument("--years", default="2024")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="data/raw_data.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # FEEDS dynamiques  + overwrite systématique
    # par défaut JSON compact (pas d'indent) + overwrite
    feeds = {str(out_path): {"format": "json", "encoding": "utf-8", "overwrite": True}}

    process = CrawlerProcess(settings={
        "FEEDS": feeds,
        "FEED_EXPORT_ENCODING": "utf-8",
        "ROBOTSTXT_OBEY": True,
        "USER_AGENT": "Mozilla/5.0 (compatible; dvf-scraper/1.0)",
        "DOWNLOAD_DELAY": 0.2,
        "CONCURRENT_REQUESTS": 2,
    })

    process.crawl(
        DVFSpider,
        departement=args.departement,
        years=args.years,
        limit=args.limit,
    )
    process.start()
    print(f"✅ Écrit (overwrite): {out_path}")
''
if __name__ == "__main__":
    _run_standalone()
