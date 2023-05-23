from sklearn.metrics import classification_report


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    return report
