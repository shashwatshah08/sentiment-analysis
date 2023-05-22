from sklearn.feature_extraction.text import TfidfVectorizer


def feature_extraction(df):
    vectorizer = TfidfVectorizer()
    text_list = df["text"].apply(lambda x: " ".join(x))
    transformed_feature_matrix = vectorizer.fit_transform(text_list)
    return transformed_feature_matrix
