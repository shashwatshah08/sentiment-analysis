# create a function to save the model
import pickle


def save_model(model):
    filename = "../models/RandomForestClassifier.sav"
    pickle.dump(model, open(filename, "wb"))


def load_model():
    try:
        filename = "../models/RandomForestClassifier.sav"
        model = pickle.load(open(filename, "rb"))
        return model
    except:
        return None
