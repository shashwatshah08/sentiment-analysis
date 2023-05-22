import re


def preprocess_data(df):
    df = remove_unwanted_columns(df)
    df["text"] = df["text"].apply(preprocess_text)
    return df


def remove_unwanted_columns(df):
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df.drop(["ids", "date", "flag", "user"], axis=1)
    return df


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z ]+", "", text)
    text = text.lower()
    return text
