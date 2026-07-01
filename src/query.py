"""
Step 3: Retrieval + Citation-Aware Generation
Retrieves top-k relevant chunks via FAISS semantic search, then prompts
Mistral to answer using only that context, with inline citations back
to source filename + page number.
"""

import os                                                    # used to read the MISTRAL_API_KEY env var
from mistralai import Mistral                                  # Mistral's official Python client
from embed_index import get_embedder, load_index, EMBED_MODEL_NAME  # reuse Step 2's embedder + index loader
from ingest import Chunk                                        # our Chunk dataclass

MISTRAL_MODEL = "mistral-large-latest"                          # the LLM used for answer generation


def retrieve(query: str, index, chunks: list[Chunk], embedder, k: int = 4) -> list[Chunk]:
    query_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    # ^ embeds the user's question the same way chunks were embedded, so they live in the same vector space
    scores, indices = index.search(query_vec, k)                  # finds the k nearest chunk vectors (highest similarity)
    return [chunks[i] for i in indices[0] if i != -1]             # maps returned indices back to actual Chunk objects


def format_context(chunks: list[Chunk]) -> str:
    """Builds a context block where each passage is labeled [n] for citation."""
    blocks = []                                                  # will hold formatted passage strings
    for i, c in enumerate(chunks, start=1):                       # number passages starting from 1 (not 0)
        blocks.append(f"[{i}] (Source: {c.source}, p.{c.page})\n{c.text}")  # tag each passage with its citation label
    return "\n\n".join(blocks)                                   # join all passages into one text block


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = format_context(chunks)                             # get the labeled passages
    return f"""You are a research assistant answering questions strictly using the provided passages.

Rules:
- Only use information found in the passages below.
- Every claim must include a citation in the form [n] referring to the passage number.
- If the passages don't contain enough information to answer, say so explicitly.

Passages:
{context}

Question: {query}

Answer (with inline [n] citations):"""
    # ^ this prompt grounds the model in retrieved context and forces it to cite passage numbers,
    #   which is what makes the output "citation-aware" rather than a free-form hallucinated answer


def generate_answer(query: str, chunks: list[Chunk], client: Mistral) -> str:
    prompt = build_prompt(query, chunks)                         # construct the full grounded prompt
    response = client.chat.complete(
        model=MISTRAL_MODEL,                                       # which Mistral model to call
        messages=[{"role": "user", "content": prompt}],             # single-turn chat message containing our prompt
        temperature=0.2,                                            # low temperature = more deterministic, less creative drift
    )
    return response.choices[0].message.content                   # extract the generated text from the API response


def format_sources(chunks: list[Chunk]) -> str:
    lines = []                                                    # will hold one line per source citation
    for i, c in enumerate(chunks, start=1):                        # match numbering used in format_context
        lines.append(f"[{i}] {c.source}, page {c.page}")             # human-readable source reference
    return "\n".join(lines)


def query_rag(query: str, index, chunks: list[Chunk], embedder, client: Mistral, k: int = 4) -> dict:
    retrieved = retrieve(query, index, chunks, embedder, k)        # step A: semantic search for relevant chunks
    answer = generate_answer(query, retrieved, client)             # step B: generate a grounded, cited answer
    return {
        "answer": answer,                                            # the model's text answer with [n] citations
        "sources": format_sources(retrieved),                         # human-readable list mapping [n] -> file/page
        "retrieved_chunks": retrieved,                                 # raw chunks, in case caller wants to inspect them
    }
