"""
Main entry point: builds the index (if needed) and starts an interactive
Q&A session over your PDFs.

Usage:
    export MISTRAL_API_KEY="your-key"
    python main.py            # build index from data/ then query
    python main.py --rebuild  # force re-ingestion + re-embedding
"""

import os                                                    # path handling, reading env vars
import sys                                                     # used to exit the script with an error code
import argparse                                                 # parses command-line flags like --rebuild
from dotenv import load_dotenv
from ingest import ingest_directory                             # Step 1 function
from embed_index import build_and_save_index, load_index, get_embedder, EMBED_MODEL_NAME, INDEX_DIR  # Step 2 functions
from query import query_rag                                     # Step 3 function
from mistralai import Mistral                                   # Mistral client for generation
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # path to the folder where you drop your PDFs



def index_exists(index_dir: str = INDEX_DIR) -> bool:
    return os.path.exists(os.path.join(index_dir, "faiss.index")) and \
           os.path.exists(os.path.join(index_dir, "chunks.pkl"))
    # ^ checks whether both required index files are already on disk, to avoid re-embedding unnecessarily


def main():
    parser = argparse.ArgumentParser()                           # sets up CLI argument parsing
    parser.add_argument("--rebuild", action="store_true", help="Force re-ingestion and re-embedding")
    # ^ if --rebuild is passed, this becomes True; otherwise False
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve per query")
    # ^ lets you control how many passages are retrieved per question from the command line
    args = parser.parse_args()                                  # actually reads sys.argv and populates args.rebuild / args.k

    if "MISTRAL_API_KEY" not in os.environ:                      # fail fast if the API key isn't set
        print("ERROR: Set MISTRAL_API_KEY environment variable first.")
        sys.exit(1)                                                 # exit code 1 signals an error to the shell

    if args.rebuild or not index_exists():                       # rebuild if forced, or if no index exists yet
        print(f"Ingesting PDFs from {DATA_DIR} ...")
        chunks = ingest_directory(DATA_DIR)                         # Step 1: extract + chunk all PDFs
        if not chunks:                                              # nothing to index if no PDFs were found/parsed
            print("No chunks produced. Add PDFs to the data/ directory and retry.")
            sys.exit(1)
        index, chunks = build_and_save_index(chunks)               # Step 2: embed chunks, build FAISS index, save to disk
    else:
        print("Loading existing index ...")
        index, chunks = load_index()                                # skip re-embedding; load the cached index from disk
        print(f"Loaded {len(chunks)} chunks.")

    embedder = get_embedder(EMBED_MODEL_NAME)                     # load the embedding model (needed to embed queries too)
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])       # initialize the Mistral API client

    print(f"\nReady. {len(chunks)} chunks indexed. Ask a question (or 'quit').\n")
    while True:                                                   # interactive loop — keeps asking until user quits
        q = input("Question: ").strip()                             # read user input, strip whitespace
        if not q:                                                    # ignore empty input, just re-prompt
            continue
        if q.lower() in ("quit", "exit"):                            # exit condition
            break
        result = query_rag(q, index, chunks, embedder, client, k=args.k)  # Step 3: retrieve + generate cited answer
        print(f"\nAnswer:\n{result['answer']}")                       # print the generated answer
        print(f"\nSources:\n{result['sources']}\n")                   # print which passages/pages were cited
        print("-" * 60)                                              # visual separator between Q&A turns


if __name__ == "__main__":                                       # only run main() when this file is executed directly
    main()                                                          # (not when imported as a module elsewhere)
