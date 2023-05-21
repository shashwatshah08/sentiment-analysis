import pandas as pd

if "__main__" == __name__:
    df = pd.read_csv(
        "./archive/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1"
    )
    print(df.head())
