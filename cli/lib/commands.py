from lib.inverted_index import InvertedIndex
from lib.search_utils import tokenize_text

from .search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def build_command():
    print("Building inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print("Inverted index built successfully")


def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_idf(term)


def tf_idf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf_idf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)


def bm25search_command(
    query: str, limit=DEFAULT_SEARCH_LIMIT
) -> list[tuple[dict, float]]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.bm25_search(query, limit)
