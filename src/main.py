import pandas as pd

from preprocess_data import remove_unwanted_columns

if "__main__" == __name__:
    df = pd.read_csv(
        "./archive/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1"
    )
    df = remove_unwanted_columns(df)
    print(df.head())
