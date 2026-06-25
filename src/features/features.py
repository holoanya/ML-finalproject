"""Tabular + spatial feature engineering.

FeatureBuilder.fit() learns everything (medians, target-encoding maps, KMeans
clusters, category lists) on the TRAIN split ONLY, then .transform() applies it to
any split. This is what keeps the held-out test honest (no leakage).

Spatial features use course methods only (no geo library): raw lat/long,
haversine distance to the city centre, and a KMeans neighbourhood cluster (L13),
target-encoded. neighbourhood_cleansed is also target-encoded.
"""
from __future__ import annotations
import ast
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

BERLIN_CENTRE = (52.520008, 13.404954)  # Brandenburg Gate-ish

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

AMENITY_FLAGS = {
    "wifi": "wifi",
    "kitchen": "kitchen",
    "ac": "air conditioning",
    "parking": "parking",
    "washer": "washer",
    "dryer": "dryer",
    "dishwasher": "dishwasher",
    "elevator": "elevator",
    "heating": "heating",
    "pool": "pool",
    "hottub": "hot tub",
    "balcony": "balcon",  # "balcony" / "balcón"
    "tv": "tv",
}

NUMERIC = [
    "accommodates", "bedrooms", "beds", "bathrooms_num", "minimum_nights",
    "availability_365", "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "calculated_host_listings_count",
    "host_response_rate_num", "host_listings_count",
]


def _haversine(lat, lon, lat0, lon0):
    R = 6371.0
    p1, p2 = np.radians(lat), np.radians(lat0)
    dphi = np.radians(lat0 - lat)
    dl = np.radians(lon0 - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _parse_bath(text):
    if pd.isna(text):
        return np.nan, 0
    t = str(text).lower()
    shared = 1 if "shared" in t else 0
    if "half" in t:
        return 0.5, shared
    m = re.search(r"([\d.]+)", t)
    return (float(m.group(1)) if m else np.nan), shared


def _amenity_count(text):
    try:
        return len(ast.literal_eval(text)) if isinstance(text, str) else 0
    except (ValueError, SyntaxError):
        return 0


def _smoothed_target_map(cat: pd.Series, y: pd.Series, smoothing: float = 10.0):
    glob = y.mean()
    agg = y.groupby(cat).agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + glob * smoothing) / (agg["count"] + smoothing)
    return smooth.to_dict(), glob


class FeatureBuilder:
    def __init__(self, n_clusters: int = 15, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed

    # ---- raw -> intermediate columns shared by fit & transform ----
    def _base_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce")
        out["bedrooms"] = pd.to_numeric(df.get("bedrooms"), errors="coerce")
        out["beds"] = pd.to_numeric(df.get("beds"), errors="coerce")
        bath = df["bathrooms_text"].apply(_parse_bath)
        out["bathrooms_num"] = [b[0] for b in bath]
        out["is_shared_bath"] = [b[1] for b in bath]
        out["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
        out["availability_365"] = pd.to_numeric(df.get("availability_365"), errors="coerce")
        out["number_of_reviews"] = pd.to_numeric(df.get("number_of_reviews"), errors="coerce")
        out["reviews_per_month"] = pd.to_numeric(df.get("reviews_per_month"), errors="coerce")
        out["review_scores_rating"] = pd.to_numeric(df.get("review_scores_rating"), errors="coerce")
        out["calculated_host_listings_count"] = pd.to_numeric(
            df.get("calculated_host_listings_count"), errors="coerce")
        out["host_listings_count"] = pd.to_numeric(df.get("host_listings_count"), errors="coerce")
        out["host_response_rate_num"] = (
            df.get("host_response_rate", pd.Series(index=df.index, dtype="object"))
            .astype("string").str.replace("%", "", regex=False)
            .replace("N/A", pd.NA).astype("float") / 100.0
        )

        # flags
        out["is_longterm"] = (out["minimum_nights"] >= 28).astype(int)
        out["has_reviews"] = (out["number_of_reviews"].fillna(0) > 0).astype(int)
        out["has_description"] = df["description"].notna().astype(int)
        out["superhost"] = (df["host_is_superhost"] == "t").astype(int)
        out["instant_bookable"] = (df["instant_bookable"] == "t").astype(int)

        # amenities
        am = df["amenities"].astype("string").fillna("")
        out["amenities_count"] = df["amenities"].apply(_amenity_count)
        am_low = am.str.lower()
        for col, kw in AMENITY_FLAGS.items():
            out[f"am_{col}"] = am_low.str.contains(kw, regex=False).astype(int)

        # room_type one-hot
        for rt in ROOM_TYPES:
            out[f"room_{rt}"] = (df["room_type"] == rt).astype(int)

        # spatial raw
        out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        out["dist_centre_km"] = _haversine(
            out["latitude"], out["longitude"], *BERLIN_CENTRE)
        return out

    def fit(self, df: pd.DataFrame, y_log: pd.Series) -> "FeatureBuilder":
        base = self._base_frame(df)
        self.medians_ = {c: base[c].median() for c in NUMERIC}

        # KMeans neighbourhood cluster on coordinates (L13), fit on train
        coords = base[["latitude", "longitude"]].to_numpy()
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        clusters = pd.Series(self.kmeans_.fit_predict(coords), index=df.index)

        # target encodings (train only)
        self.te_neigh_, self.te_neigh_glob_ = _smoothed_target_map(
            df["neighbourhood_cleansed"], y_log)
        self.te_clust_, self.te_clust_glob_ = _smoothed_target_map(clusters, y_log)
        self.te_prop_, self.te_prop_glob_ = _smoothed_target_map(
            df["property_type"], y_log)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._base_frame(df)
        for c in NUMERIC:
            base[c] = base[c].fillna(self.medians_[c])

        clusters = pd.Series(
            self.kmeans_.predict(base[["latitude", "longitude"]].to_numpy()),
            index=df.index)
        base["te_neighbourhood"] = (
            df["neighbourhood_cleansed"].map(self.te_neigh_).fillna(self.te_neigh_glob_))
        base["te_cluster"] = clusters.map(self.te_clust_).fillna(self.te_clust_glob_)
        base["te_property_type"] = (
            df["property_type"].map(self.te_prop_).fillna(self.te_prop_glob_))
        return base.astype(float)


def build_features(train_df, test_df, y_train) -> tuple[pd.DataFrame, pd.DataFrame]:
    fb = FeatureBuilder().fit(train_df, y_train)
    return fb.transform(train_df), fb.transform(test_df)
