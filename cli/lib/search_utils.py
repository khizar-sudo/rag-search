import json
import os
import string

from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()

    stop_words = load_stop_words()

    stemmer = PorterStemmer()

    valid_tokens = []
    for token in tokens:
        if token and token not in stop_words:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens
