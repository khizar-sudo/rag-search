#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
)
from lib.semantic_search import (
    chunk_text,
    embed_query_text,
    embed_text,
    search_command,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify
    subparsers.add_parser("verify", help="Verify the semantic search model")

    # Embed
    embed_parser = subparsers.add_parser("embed_text", help="Embed a text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    # Verify embeddings
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    # Embed query text
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    # Search
    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, help="Search limit", default=DEFAULT_SEARCH_LIMIT
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, help="Chunk size", default=DEFAULT_CHUNK_SIZE
    )
    chunk_parser.add_argument(
        "--overlap", type=int, help="Chunk overlap", default=DEFAULT_CHUNK_OVERLAP
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
