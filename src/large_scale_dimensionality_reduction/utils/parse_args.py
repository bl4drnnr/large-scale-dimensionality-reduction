import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text Embedding Generator for Datasets and VectorDB Storage",
        epilog="""
        Przykłady użycia:

        python3 main.py --mode upload --model all-MiniLM-L6-v2 --dataset data/goemotions_1.csv data/goemotions_2.csv --collection goemotions_embeddings

        python3 main.py --mode query --model all-MiniLM-L6-v2 --collection goemotions_embeddings --query "I love sunny days" --top-k 5
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["upload", "query"],
        help="Mode: 'upload' to upload data, 'query' to search similar texts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the SentenceTransformer model to use.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Paths to one or more CSV files containing text data.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="text_embeddings",
        help="Name of the collection to store embeddings in ChromaDB.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query text to search for similar documents (required for query mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similar results to return for a query (default: 5).",
    )
    return parser.parse_args()
