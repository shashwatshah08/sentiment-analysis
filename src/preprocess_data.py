import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')


def preprocess_data(df):
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df.drop(["ids", "date", "flag", "user"], axis=1)
    df["text"] = df["text"].apply(preprocess_text)
    df["text"] = df["text"].apply(nltk.word_tokenize)
    stop_words = set(stopwords.words("english"))
    df["text"] = df["text"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    lemmatizer = WordNetLemmatizer()
    df["text"] = df["text"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return df


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z ]+", "", text)
    text = text.lower()
    return text
