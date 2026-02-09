import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

DATA_PATH = "database/myopia.csv"

def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    return df

#Testing if data loads
def inspect_data(df):
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)
    print("\nFirst 5 rows:")
    print(df)


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)

def prepare_data(df):
    df = df.copy()
    # relevant columns for now
    cols = [
        "AGE",
        "GENDER",
        "SPHEQ",
        "MOMMY",
        "DADMY",
        "TVHR",
        "READHR",
        "SPORTHR"
        
    ]
    df = df[cols]
    #Convert to num
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # if != spheq row is useless 
    df = df.dropna(subset=["SPHEQ"])
    

    df["SCREEN_TIME"] = df["TVHR"]+ 0.5 * df["READHR"]
    df["OUTDOOR_TIME"] = df["SPORTHR"]
    df["AGE_x_SCREEN"] = df["AGE"] * df["SCREEN_TIME"]
    df["AGE_x_OUTDOOR"] = df["AGE"] * df["OUTDOOR_TIME"]
    df["GENETIC_RISK"] = df["MOMMY"] + df["DADMY"]
    return df


def get_clean_data():
    df = load_data()
    df = prepare_data(df)
    return df

if __name__ == "__main__":
    df_raw = load_data()
    inspect_data(df_raw)

    df_clean = prepare_data(df_raw)
    print("\nCleaned data preview:")
    print(df_clean.head())


"""
#dummy
def predict_dummy(age, mommy, dadmy):
    ages = list(range(age, 109))
    # fake linear progression (placeholder)
    spheq = [-0.2 * (a - age) for a in ages]

    return {
        "ages": ages,
        "spheq": spheq
    }
"""


def train_model(df):
    X = df[["AGE","SCREEN_TIME", "OUTDOOR_TIME", "AGE_x_SCREEN", "AGE_x_OUTDOOR", "GENETIC_RISK", "GENDER"]]
    Y = df["SPHEQ"]
    model = LinearRegression()
    model.fit(X,Y)
    return model

def progression_tracker(age, gender, mommy, dadmy, screen_time, outdoor_time):
    df = get_clean_data()
    model = train_model(df)
    end_age = 25
    ages = list(range(age,end_age + 1))
    n = len(ages)
    X_pred = pd.DataFrame({
        "AGE" : ages,
        "SCREEN_TIME" : [screen_time] * n,
        "OUTDOOR_TIME" : [outdoor_time] * n,
        "AGE_x_SCREEN": [a * screen_time for a in ages] ,
        "AGE_x_OUTDOOR": [a * outdoor_time for a in ages] ,
        "GENETIC_RISK": [mommy + dadmy] * n,
        
        "GENDER": [gender] * n,

    })

    spheq_pred = model.predict(X_pred)
    delta = spheq_pred - spheq_pred[0]


    return{
        "ages" : ages,
        "delta" : delta.tolist()
    }