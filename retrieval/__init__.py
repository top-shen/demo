"""Train-only text retrieval support for RI-VerbalTS."""

from .text_retriever import (
    INDEX_VERSION,
    LongCLIPTextEmbedder,
    TextRetriever,
    build_retrieval_index,
    load_retrieval_index,
)

__all__ = [
    "INDEX_VERSION",
    "LongCLIPTextEmbedder",
    "TextRetriever",
    "build_retrieval_index",
    "load_retrieval_index",
]
