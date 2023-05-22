def remove_unwanted_columns(df):
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df.drop(["ids", "date", "flag", "user"], axis=1)
    return df
