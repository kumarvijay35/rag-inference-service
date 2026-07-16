"""Paragraph-aware text chunking with overlap.

Deliberately implemented by hand (no LangChain dependency in this service).
Strategy:
  1. Split on paragraph boundaries first, so chunks respect natural structure.
  2. Pack paragraphs into chunks up to `chunk_size` characters.
  3. If a single paragraph exceeds chunk_size, hard-split it on sentence-ish
     boundaries.
  4. Carry `chunk_overlap` characters of trailing context into the next chunk
     so answers spanning a chunk boundary are still retrievable.
"""

import re


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split an oversized block, preferring sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # sentence itself longer than chunk_size -> slice it
            while len(sentence) > chunk_size:
                chunks.append(sentence[:chunk_size])
                sentence = sentence[chunk_size - overlap :]
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(para, chunk_size, chunk_overlap))
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            chunks.append(current)
            # overlap: seed next chunk with the tail of the previous one
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    return chunks
