import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

DATASET_PATH = "/coding/data/unique_tweets_7k.csv"

def load_dataset(path):
    data = pd.read_csv(path)
    tweet = data.drop_duplicates(subset=["text"])
    tweet = tweet[["text", "sentiment"]].dropna()
    return tweet


def data_cleaning(df):
    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_html(text):
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)

    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_punct(text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    def remove_multi_spaces(text):
        space = re.compile(' +')
        line = re.compile('\n')
        return space.sub(r' ', line.sub(r' ', text))

    def remove_hashtags_mentions(text):
        hashtags = re.compile(r"^#\S+|\s#\S+")
        mentions = re.compile(r"^@\S+|\s@\S+")
        text = hashtags.sub(' hashtag', text)
        text = mentions.sub(' entity', text)
        return text.strip().lower()

    df.text = df.text.apply(lambda x: remove_URL(x))
    df.text = df.text.apply(lambda x: remove_html(x))
    df.text = df.text.apply(lambda x: remove_emoji(x))
    df.text = df.text.apply(lambda x: remove_punct(x))
    df.text = df.text.apply(lambda x: remove_multi_spaces(x))
    df.text = df.text.apply(lambda x: remove_hashtags_mentions(x))
    return df


def balance_data(df):
    df = df.drop(df.query('sentiment == 0').sample(frac=0.7).index)
    df = df.drop(df.query('sentiment == 4').sample(frac=0.6).index)
    #df = df[df["sentiment"] != 3]
    #df.loc[df['sentiment'] == 4, "sentiment"] = 3
    return df


def set_split(df, test_size = 0.2):
    train_val, test = train_test_split(df, test_size = test_size, random_state = 42)
    return train_val, test


def prepare_train_test_from_file(path):
    tweets = load_dataset(path)
    tweets = data_cleaning(tweets)
    tweets = balance_data(tweets)
    return set_split(tweets)


if __name__ == "main":
    tweets = load_dataset(DATASET_PATH)
    tweets = data_cleaning(tweets)
    tweets = balance_data(tweets)
    train_val, test = set_split(tweets)