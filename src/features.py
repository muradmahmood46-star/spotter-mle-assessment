"""
Feature engineering and transformation pipeline for Freight Rate Prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    miles = 3958.8 * c
    return miles


class FeaturePipeline:
    def __init__(self):
        self.pickup_coords = {}
        self.delivery_coords = {}
        self.equipment_weight_median = {}
        self.overall_weight_median = 30000.0
        self.mean_market_index = 1.0
        self.mean_quote_signal = 2.0
        self.equipment_map = {}
        self.pickup_map = {}
        self.delivery_map = {}
        self.lane_map = {}

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        # Build coordinates map by city
        for _, row in train_df.iterrows():
            if row["pickup"] not in self.pickup_coords and pd.notna(row.get("pickup_lat")):
                self.pickup_coords[row["pickup"]] = (row["pickup_lat"], row["pickup_lon"])
            if row["delivery"] not in self.delivery_coords and pd.notna(row.get("delivery_lat")):
                self.delivery_coords[row["delivery"]] = (row["delivery_lat"], row["delivery_lon"])

        # Weight medians
        self.equipment_weight_median = train_df.groupby("equipment")["weight"].median().to_dict()
        self.overall_weight_median = train_df["weight"].median()

        # Market & quote signal means
        self.mean_market_index = train_df["market_index"].mean()
        self.mean_quote_signal = train_df["quote_signal"].mean()

        # Categorical mappings
        equipments = sorted(train_df["equipment"].dropna().unique())
        self.equipment_map = {eq: idx for idx, eq in enumerate(equipments)}

        pickups = sorted(train_df["pickup"].dropna().unique())
        self.pickup_map = {p: idx for idx, p in enumerate(pickups)}

        deliveries = sorted(train_df["delivery"].dropna().unique())
        self.delivery_map = {d: idx for idx, d in enumerate(deliveries)}

        lanes = sorted((train_df["pickup"] + "_" + train_df["delivery"]).unique())
        self.lane_map = {l: idx for idx, l in enumerate(lanes)}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])

        # 1. Fill missing coordinates if not present (e.g., december chart inputs)
        if "pickup_lat" not in data.columns:
            data["pickup_lat"] = data["pickup"].map(lambda p: self.pickup_coords.get(p, (np.nan, np.nan))[0])
            data["pickup_lon"] = data["pickup"].map(lambda p: self.pickup_coords.get(p, (np.nan, np.nan))[1])
        if "delivery_lat" not in data.columns:
            data["delivery_lat"] = data["delivery"].map(lambda d: self.delivery_coords.get(d, (np.nan, np.nan))[0])
            data["delivery_lon"] = data["delivery"].map(lambda d: self.delivery_coords.get(d, (np.nan, np.nan))[1])

        # 2. Impute missing market index and quote signal if missing
        if "market_index" not in data.columns:
            data["market_index"] = self.mean_market_index
        else:
            data["market_index"] = data["market_index"].fillna(self.mean_market_index)

        if "quote_signal" not in data.columns:
            data["quote_signal"] = self.mean_quote_signal
        else:
            data["quote_signal"] = data["quote_signal"].fillna(self.mean_quote_signal)

        # 3. Impute weight
        def fill_weight(row):
            if pd.notna(row.get("weight")):
                return row["weight"]
            eq = row.get("equipment")
            return self.equipment_weight_median.get(eq, self.overall_weight_median)

        data["weight_imputed"] = data.apply(fill_weight, axis=1)

        # 4. Spatial / Distance Features
        data["haversine_dist"] = haversine_np(
            data["pickup_lat"], data["pickup_lon"],
            data["delivery_lat"], data["delivery_lon"]
        )
        data["lat_diff"] = (data["delivery_lat"] - data["pickup_lat"]).abs()
        data["lon_diff"] = (data["delivery_lon"] - data["pickup_lon"]).abs()
        data["dist_ratio"] = data["distance"] / (data["haversine_dist"] + 1.0)

        # 5. Temporal Features
        data["dayofweek"] = data["date"].dt.dayofweek
        data["day"] = data["date"].dt.day
        data["month"] = data["date"].dt.month
        data["dayofyear"] = data["date"].dt.dayofyear
        data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
        data["sin_dayofweek"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
        data["cos_dayofweek"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
        data["sin_dayofyear"] = np.sin(2 * np.pi * data["dayofyear"] / 365.25)
        data["cos_dayofyear"] = np.cos(2 * np.pi * data["dayofyear"] / 365.25)

        # 6. Interaction Features
        data["market_dist"] = data["market_index"] * data["distance"]
        data["quote_dist"] = data["quote_signal"] * data["distance"]
        data["quote_market"] = data["quote_signal"] * data["market_index"]
        data["weight_dist"] = data["weight_imputed"] * data["distance"] / 10000.0

        # 7. Categorical Encodings
        data["equipment_cat"] = data["equipment"].map(self.equipment_map).fillna(-1).astype(int)
        data["pickup_cat"] = data["pickup"].map(self.pickup_map).fillna(-1).astype(int)
        data["delivery_cat"] = data["delivery"].map(self.delivery_map).fillna(-1).astype(int)
        lane_str = data["pickup"] + "_" + data["delivery"]
        data["lane_cat"] = lane_str.map(self.lane_map).fillna(-1).astype(int)

        feature_cols = [
            "distance", "weight_imputed", "market_index", "quote_signal",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "haversine_dist", "lat_diff", "lon_diff", "dist_ratio",
            "dayofweek", "day", "month", "dayofyear", "is_weekend",
            "sin_dayofweek", "cos_dayofweek", "sin_dayofyear", "cos_dayofyear",
            "market_dist", "quote_dist", "quote_market", "weight_dist",
            "equipment_cat", "pickup_cat", "delivery_cat", "lane_cat"
        ]

        return data[feature_cols].fillna(0)
