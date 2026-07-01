"""
Step 1: PDF Ingestion & Chunking
Extracts text from PDFs using PyMuPDF, splits into ~400-token chunks
using LangChain's text splitter, and tags each chunk with source metadata
for citation tracking later.
"""

import os                                                  # used to list files in a directory and join paths
import fitz                                                 # PyMuPDF — fast PDF parsing/text extraction library
from langchain.text_splitter import RecursiveCharacterTextSplitter  # LangChain's smart text chunker
from dataclasses import dataclass                            # lets us define a simple typed data container


@dataclass                                                  # auto-generates __init__, __repr__ etc. for this class
class Chunk:
    text: str                                                # the actual chunk text content
    source: str                                              # filename the chunk came from (for citations)
    page: int                                                # page number within that PDF (1-indexed)
    chunk_id: int                                            # sequential index of this chunk within the document


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number, text) using PyMuPDF."""
    doc = fitz.open(pdf_path)                                 # opens the PDF file and loads its structure
    pages = []                                                 # will hold (page_number, text) tuples
    for i, page in enumerate(doc):                              # iterate over every page; i is 0-indexed
        text = page.get_text("text")                             # extract plain text from this page
        if text.strip():                                          # skip pages that are blank/whitespace only
            pages.append((i + 1, text))                            # store as 1-indexed page number + text
    doc.close()                                                # release the file handle / free memory
    return pages


def chunk_document(pdf_path: str, chunk_size_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    """
    Splits a single PDF into ~400-token chunks, preserving page-level
    provenance for each chunk (needed for citations).

    Note: we approximate tokens as ~4 chars/token (rough but standard
    heuristic), so chunk_size_tokens=400 -> ~1600 chars.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens * 4,                         # convert token target to char target (~4 chars/token)
        chunk_overlap=overlap_tokens * 4,                         # overlap between consecutive chunks, in chars
        separators=["\n\n", "\n", ". ", " ", ""],                 # tries paragraph, then line, then sentence, then word
    )

    filename = os.path.basename(pdf_path)                        # strip directory path, keep just the filename
    pages = extract_pages(pdf_path)                              # get list of (page_number, text) for this PDF
    chunks = []                                                  # will hold Chunk objects for this document
    chunk_id = 0                                                 # running counter for chunk_id within this doc

    for page_num, page_text in pages:                            # process each page's text separately
        splits = splitter.split_text(page_text)                    # break this page's text into ~400-token pieces
        for s in splits:                                           # for every resulting text piece
            chunks.append(Chunk(text=s, source=filename, page=page_num, chunk_id=chunk_id))  # wrap it with metadata
            chunk_id += 1                                            # increment chunk id for the next piece

    return chunks


def ingest_directory(data_dir: str, chunk_size_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    """Ingest every PDF in a directory and return a flat list of chunks."""
    all_chunks = []                                              # accumulates chunks across all PDFs in the folder
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]  # filter listing to .pdf files only

    if not pdf_files:                                            # guard clause: nothing to do if no PDFs found
        print(f"No PDFs found in {data_dir}")
        return all_chunks

    for fname in pdf_files:                                      # process each PDF file one at a time
        path = os.path.join(data_dir, fname)                       # build the full file path
        try:
            doc_chunks = chunk_document(path, chunk_size_tokens, overlap_tokens)  # chunk this single PDF
            all_chunks.extend(doc_chunks)                            # add its chunks to the master list
            print(f"  {fname}: {len(doc_chunks)} chunks")             # log progress per file
        except Exception as e:                                     # catch corrupt/unreadable PDFs without crashing
            print(f"  Failed to process {fname}: {e}")

    print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")  # final summary
    return all_chunks
