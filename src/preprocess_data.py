import re


def preprocess_data(df):
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df.drop(["ids", "date", "flag", "user"], axis=1)
    df["text"] = df["text"].apply(preprocess_text)
    return df


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z ]+", "", text)
    text = text.lower()
    return text
