"""Download the pinned Berlin Inside Airbnb snapshot into data/raw/.

Reproducible: everyone runs this to get the exact same files instead of
sharing data through a cloud drive. Edit CITY_BASE to change the snapshot.
"""
from pathlib import Path
import urllib.request

CITY_BASE = "https://data.insideairbnb.com/germany/be/berlin/2025-09-23"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

FILES = {
    "listings.csv.gz": f"{CITY_BASE}/data/listings.csv.gz",
    "neighbourhoods.geojson": f"{CITY_BASE}/visualisations/neighbourhoods.geojson",
    # uncomment if/when the text-from-reviews modality is added:
    # "reviews.csv.gz": f"{CITY_BASE}/data/reviews.csv.gz",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists():
            print(f"skip (exists): {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    print("done.")


if __name__ == "__main__":
    main()
