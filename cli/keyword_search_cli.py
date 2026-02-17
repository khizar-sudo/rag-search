#!/usr/bin/env python3
import argparse

from lib.commands import build_command, search_command, tf_command
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get the term frequency of a term in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate[dict](results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            build_command()
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"TF for {args.term} in document {args.doc_id} = {tf}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
