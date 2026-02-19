import json
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np

from lib.search_utils import (
    CACHE_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_SEARCH_LIMIT,
    EMBEDDINGS_PATH,
    SEMANTIC_CHUNK_REGEX,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def generate_embedding(self, text: str) -> list[float]:
        if len(text) == 0 or text.isspace():
            raise ValueError("Text cannot be empty or whitespace")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.documents_map = {doc["id"]: doc for doc in documents}
        self.embeddings = self.model.encode(
            [f"{doc['title']} {doc['description']}" for doc in documents],
            show_progress_bar=True,
        )
        os.makedirs(CACHE_PATH, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.documents_map = {doc["id"]: doc for doc in documents}

        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(self.documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        if self.documents is None:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )
        if not self.documents_map:
            raise ValueError(
                "No documents map loaded. Call `load_or_create_embeddings` first."
            )
        if len(self.embeddings) != len(self.documents):
            raise ValueError(
                "Embeddings and documents do not match. Call `load_or_create_embeddings` first."
            )
        if len(self.documents_map) != len(self.documents):
            raise ValueError(
                "Documents map and documents do not match. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarities = [
            cosine_similarity(query_embedding, embedding)
            for embedding in self.embeddings
        ]
        results = sorted(
            zip(similarities, self.documents), key=lambda x: x[0], reverse=True
        )
        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in results[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.documents_map = {doc["id"]: doc for doc in documents}

        chunks = []
        metadata = []
        for doc in documents:
            if doc["description"] is not None and doc["description"] != "":
                doc_chunks = semantic_chunking(doc["description"], 4, 1)
                chunks.extend(doc_chunks)
                metadata.extend(
                    [
                        {
                            "movie_idx": doc["id"],
                            "chunk_idx": i,
                            "total_chunks": len(doc_chunks),
                        }
                        for i in range(len(doc_chunks))
                    ]
                )

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        os.makedirs(CACHE_PATH, exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.documents_map = {doc["id"]: doc for doc in documents}

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
                self.chunk_metadata = chunk_data["chunks"]

            doc_ids_in_cache = {m["movie_idx"] for m in self.chunk_metadata}
            doc_ids_current = {doc["id"] for doc in documents}
            if doc_ids_in_cache == doc_ids_current:
                return self.chunk_embeddings

        return self.build_chunk_embeddings(self.documents)


def verify_model():
    print("Verifying model...")
    semantic_search = SemanticSearch()
    print("Model loaded:", semantic_search.model)
    print("Max sequence length:", semantic_search.model.max_seq_length)


def embed_text(text: str) -> list[float]:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {len(embedding)}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> list[float]:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['title']} (score: {result['score']:.2f})\n {result['description']}\n"
        )


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks = [words[i : i + chunk_size] for i in range(0, len(words), step)]
    return [" ".join(chunk) for chunk in chunks]


def chunk_command(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semantic_chunking(
    text: str,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(SEMANTIC_CHUNK_REGEX, text)

    step = max(1, max_chunks - overlap)
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = sentences[i : min(len(sentences), i + max_chunks)]
        chunks.append(" ".join(chunk))

        if i + max_chunks >= len(sentences):
            break
    return chunks


def semantic_chunk_command(
    text: str,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    chunks = semantic_chunking(text, max_chunks, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_chunks():
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(
        f"Generated {len(chunked_semantic_search.chunk_embeddings)} chunked embeddings"
    )
