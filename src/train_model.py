from sklearn.ensemble import RandomForestClassifier


def train_model(x_train, y_train):
    model = RandomForestClassifier(n_jobs=4)
    model.fit(x_train, y_train)
    return model
