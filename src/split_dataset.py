from sklearn.model_selection import train_test_split


def split_dataset(features_matrix, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        features_matrix, labels, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test
