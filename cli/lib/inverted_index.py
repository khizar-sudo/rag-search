import math

from collections import Counter, defaultdict
import pickle
import os

from lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT, tokenize_text
from lib.search_utils import CACHE_PATH, load_movies


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict[str, set[int]](set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict[int, Counter[str]](Counter)
        self.doc_lengths = defaultdict[int, int](int)

        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize_text(text)
        self.term_frequencies[doc_id].update(tokenized_text)
        for word in set[str](tokenized_text):
            self.index[word].add(doc_id)
        self.doc_lengths[doc_id] = len(tokenized_text)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single word")
        doc_ids = self.index.get(tokens[0], set())
        return sorted(list[int](doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single word")
        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single word")
        return math.log(len(self.docmap) / (len(self.get_documents(tokens[0])) + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single word")

        doc_freq = len(self.get_documents(tokens[0]))
        total_docs = len(self.docmap)

        return math.log(((total_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        avg_doc_length = self.__get_avg_doc_length()
        length_normalization = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        bm25_saturation = (tf * (k1 + 1)) / (tf + k1 * length_normalization)
        return bm25_saturation

    def bm25(self, doc_id, term):
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)

    def bm25_search(
        self, query, limit=DEFAULT_SEARCH_LIMIT
    ) -> list[tuple[dict, float]]:
        tokens = tokenize_text(query)
        scores = defaultdict[int, float](float)
        for token in tokens:
            for doc_id in self.index[token]:
                scores[doc_id] += self.bm25(doc_id, token)

        # Return top N documents along with their BM25 scores
        return sorted(
            [(self.docmap[doc_id], score) for doc_id, score in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if (
            not os.path.exists(self.index_path)
            or not os.path.exists(self.docmap_path)
            or not os.path.exists(self.term_frequencies_path)
            or not os.path.exists(self.doc_lengths_path)
        ):
            raise FileNotFoundError(
                "Inverted index files not found. Please run the 'build' command first."
            )

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
