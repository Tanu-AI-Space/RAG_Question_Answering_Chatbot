"""
Step 2: Embedding & FAISS Indexing
Embeds chunks using Sentence Transformers and builds a FAISS index
for fast semantic similarity search. Also persists the index + chunk
metadata to disk so we don't have to re-embed every run.
"""

import os                                                    # file/path handling
import pickle                                                 # used to serialize the list of Chunk objects to disk
import numpy as np                                             # numerical arrays required by FAISS
import faiss                                                   # Facebook AI Similarity Search — vector index library
from sentence_transformers import SentenceTransformer            # loads pretrained embedding models
from ingest import Chunk                                        # our Chunk dataclass from Step 1

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, 384-dim, strong general-purpose sentence embedding model
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")  # where the FAISS index + chunks are persisted


def get_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)                       # downloads (first time) and loads the model into memory


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    texts = [c.text for c in chunks]                              # pull out just the text strings to embed
    embeddings = model.encode(
        texts,
        batch_size=batch_size,                                      # process this many chunks per forward pass (speed/memory tradeoff)
        show_progress_bar=True,                                     # prints a progress bar since this can take a while for many chunks
        convert_to_numpy=True,                                      # return numpy arrays instead of torch tensors
        normalize_embeddings=True,                                  # scales each vector to unit length, so L2 distance ~ cosine similarity
    )
    return embeddings.astype("float32")                           # FAISS requires float32, not float64


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]                                    # embedding dimensionality (384 for this model)
    index = faiss.IndexFlatIP(dim)                                # "Inner Product" index — equals cosine similarity since vectors are normalized
    index.add(embeddings)                                        # load all vectors into the index for searching
    return index


def save_index(index: faiss.Index, chunks: list[Chunk], out_dir: str = INDEX_DIR):
    os.makedirs(out_dir, exist_ok=True)                          # create the index/ folder if it doesn't exist
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))  # serialize the FAISS index to disk
    with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:      # open a binary file for writing
        pickle.dump(chunks, f)                                        # serialize the Chunk objects (text + metadata)
    print(f"Saved index ({index.ntotal} vectors) and {len(chunks)} chunks to {out_dir}")


def load_index(in_dir: str = INDEX_DIR) -> tuple[faiss.Index, list[Chunk]]:
    index = faiss.read_index(os.path.join(in_dir, "faiss.index"))  # load the saved FAISS index back into memory
    with open(os.path.join(in_dir, "chunks.pkl"), "rb") as f:        # open the pickled chunks file
        chunks = pickle.load(f)                                        # restore the list of Chunk objects
    return index, chunks


def build_and_save_index(chunks: list[Chunk], model_name: str = EMBED_MODEL_NAME):
    model = get_embedder(model_name)                              # load the embedding model
    print(f"Embedding {len(chunks)} chunks with {model_name}...")
    embeddings = embed_chunks(chunks, model)                       # convert all chunk texts into vectors
    index = build_faiss_index(embeddings)                          # build the searchable FAISS index from those vectors
    save_index(index, chunks)                                      # persist both index and chunk metadata to disk
    return index, chunks
