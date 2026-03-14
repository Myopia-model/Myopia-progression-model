import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(path=None):
    """Load and return raw myopia CSV data."""
    if path is None:
        path = os.path.join(BASE_DIR, "myopia.csv")
    df = pd.read_csv(path, sep=";")
    return df


def inspect_data(df):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())


def prepare_data(df):
    df = df.copy()
    cols = [
        "AGE", "GENDER", "SPHEQ",
        "MOMMY", "DADMY",
        "TVHR", "READHR", "COMPHR", "STUDYHR", "SPORTHR"
    ]
    # Only keep cols that exist in df
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["SPHEQ"])

    # Near-work: TV + reading + computer + studying
    df["SCREEN_TIME"] = (
        df.get("TVHR", pd.Series(0, index=df.index)).fillna(0)
        + df.get("COMPHR", pd.Series(0, index=df.index)).fillna(0)
        + 0.5 * df.get("READHR", pd.Series(0, index=df.index)).fillna(0)
        + 0.5 * df.get("STUDYHR", pd.Series(0, index=df.index)).fillna(0)
    )
    df["OUTDOOR_TIME"] = df.get("SPORTHR", pd.Series(0, index=df.index)).fillna(0)
    df["AGE_x_SCREEN"] = df["AGE"] * df["SCREEN_TIME"]
    df["AGE_x_OUTDOOR"] = df["AGE"] * df["OUTDOOR_TIME"]
    df["GENETIC_RISK"] = (
        df.get("MOMMY", pd.Series(0, index=df.index)).fillna(0)
        + df.get("DADMY", pd.Series(0, index=df.index)).fillna(0)
    )
    return df


def get_clean_data(path=None):
    df = load_data(path)
    df = prepare_data(df)
    return df


def train_model(df):
    features = ["AGE", "SCREEN_TIME", "OUTDOOR_TIME",
                 "AGE_x_SCREEN", "AGE_x_OUTDOOR", "GENETIC_RISK", "GENDER"]
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)
    y = df["SPHEQ"]
    model = LinearRegression()
    model.fit(X, y)
    return model, features


def progression_tracker(age, gender, mommy, dadmy, screen_time, outdoor_time, data_path=None):
    """
    Predict myopia progression from `age` to 25.

    Returns dict with:
      - ages: list of ages
      - spheq_pred: absolute predicted SPHEQ values
      - delta: change from baseline (age 0 = 0)
      - baseline_spheq: predicted SPHEQ at the given age
    """
    df = get_clean_data(data_path)
    model, features = train_model(df)

    end_age = 25
    ages = list(range(age, end_age + 1))
    n = len(ages)

    X_pred = pd.DataFrame({
        "AGE": ages,
        "SCREEN_TIME": [screen_time] * n,
        "OUTDOOR_TIME": [outdoor_time] * n,
        "AGE_x_SCREEN": [a * screen_time for a in ages],
        "AGE_x_OUTDOOR": [a * outdoor_time for a in ages],
        "GENETIC_RISK": [mommy + dadmy] * n,
        "GENDER": [gender] * n,
    })
    # Only keep features used during training
    X_pred = X_pred[[f for f in features if f in X_pred.columns]]

    spheq_pred = model.predict(X_pred)
    delta = spheq_pred - spheq_pred[0]

    return {
        "ages": ages,
        "spheq_pred": spheq_pred.tolist(),
        "delta": delta.tolist(),
        "baseline_spheq": float(spheq_pred[0]),
    }


def get_risk_label(genetic_risk, screen_time, outdoor_time):
    """Simple heuristic risk tier for display."""
    score = 0
    if genetic_risk >= 2:
        score += 2
    elif genetic_risk == 1:
        score += 1
    if screen_time >= 6:
        score += 2
    elif screen_time >= 3:
        score += 1
    if outdoor_time < 1:
        score += 1

    if score >= 4:
        return "High", "#ef4444"
    elif score >= 2:
        return "Moderate", "#f59e0b"
    else:
        return "Low", "#22c55e"


if __name__ == "__main__":
    df_raw = load_data()
    inspect_data(df_raw)
    df_clean = prepare_data(df_raw)
    print("\nCleaned data preview:")
    print(df_clean.head())

    result = progression_tracker(
        age=8, gender=1, mommy=1, dadmy=0,
        screen_time=4, outdoor_time=1
    )
    print("\nProgression result (ages):", result["ages"][:5])
    print("SPHEQ pred (first 5):", result["spheq_pred"][:5])
    print("Delta (first 5):", result["delta"][:5])