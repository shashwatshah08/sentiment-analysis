import pandas as pd

from preprocess_data import preprocess_data

if "__main__" == __name__:
    print("loading data...")
    df = pd.read_csv(
        "./archive/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1"
    )
    print("data loaded")
    print("preprocessing data...")
    df = preprocess_data(df)
    print("data preprocessed")
    print(df.head())
