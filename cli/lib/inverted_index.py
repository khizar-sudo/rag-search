from collections import Counter, defaultdict
import pickle
import os

from lib.search_utils import tokenize_text
from lib.search_utils import CACHE_PATH, load_movies


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize_text(text)
        for word in set[str](tokenized_text):
            self.index[word].add(doc_id)
            self.term_frequencies[doc_id][word] += 1

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list[int](doc_ids))

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise FileNotFoundError("Inverted index files not found")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
