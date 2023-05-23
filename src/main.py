import time

import pandas as pd

from preprocess_data import preprocess_data
from src.feature_extraction import feature_extraction
from src.save_model import save_model, load_model
from src.split_dataset import split_dataset
from src.train_model import train_model

if "__main__" == __name__:
    print("loading model...")
    model = load_model()
    if not model:
        print("model not found")
        print("loading data...")
        df = pd.read_csv(
            "../archive/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1"
        )
        print("data loaded")
        print("preprocessing data...")
        df = preprocess_data(df)
        print("data preprocessed")
        print("creating feature matrix...")
        matrix = feature_extraction(df)
        print("feature matrix created")
        print("splitting dataset...")
        x_train, x_test, y_train, y_test = split_dataset(matrix, df["target"])
        print("dataset splitted")
        print("training model...")
        start = time.time()
        model = train_model(x_train, y_train)
        end = time.time()
        print("model trained. Time -> ", end - start)
        print("saving model...")
        save_model(model)
        print("model saved")
    else:
        print("model loaded")
