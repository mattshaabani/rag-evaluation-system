"""
src/data/chunker.py

Three chunking strategies for the RAG pipeline:
  1. SlidingWindowChunker  — fixed size with overlap
  2. SemanticChunker       — splits on meaning shifts using embeddings
  3. HierarchicalChunker   — multi-level: section → paragraph

All chunkers take List[Document] and return List[Chunk].
A Chunk is a Document with additional chunking metadata.

Usage:
    from src.data.chunker import SlidingWindowChunker, SemanticChunker, HierarchicalChunker
    from src.data.loader import DocumentLoader

    loader  = DocumentLoader()
    docs    = loader.load("data/raw/paper.pdf")

    chunker = SlidingWindowChunker(chunk_size=512, chunk_overlap=50)
    chunks  = chunker.chunk(docs)
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import numpy as np

from src.data.loader import Document
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 1. Chunk dataclass (extends Document concept)
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single chunk of text ready for embedding and storage.

    Inherits the concept of Document but adds chunking metadata:
    - which chunk number within its parent document
    - which strategy created it
    - for hierarchical: which parent chunk it belongs to
    """
    content:  str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"Chunk("
            f"chars={len(self.content)}, "
            f"strategy={self.metadata.get('strategy', '?')}, "
            f"preview='{preview}...')"
        )

    @property
    def token_count(self) -> int:
        """
        Approximate token count.
        Rule of thumb: 1 token ≈ 4 characters in English.
        This avoids importing a tokenizer just for counting.
        For exact counts you'd use tiktoken.
        """
        return len(self.content) // 4


# ─────────────────────────────────────────────
# 2. Token-aware text splitter (used by Strategy 1)
# ─────────────────────────────────────────────

def split_into_tokens(text: str) -> list[str]:
    """
    Split text into word-level tokens.
    We use words as a proxy for tokens — close enough for chunking.
    In production you'd use tiktoken for exact BPE token counts.
    """
    return text.split()


def tokens_to_text(tokens: list[str]) -> str:
    return " ".join(tokens)


# ─────────────────────────────────────────────
# 3. Strategy 1: Sliding Window Chunker
# ─────────────────────────────────────────────

class SlidingWindowChunker:
    """
    Splits text into fixed-size chunks with overlap.

    The algorithm:
        tokens = split document into words
        i = 0
        while i < len(tokens):
            chunk = tokens[i : i + chunk_size]
            yield chunk
            i += (chunk_size - chunk_overlap)   ← step size

    The step size is (chunk_size - overlap), NOT chunk_size.
    This is what creates the overlap between consecutive chunks.

    Example with chunk_size=6, overlap=2:
        tokens:  [A B C D E F G H I J]
        chunk 1: [A B C D E F]
        chunk 2:         [E F G H I J]   ← E,F repeated from chunk 1
                     ↑ overlap = 2
    """

    def __init__(
        self,
        chunk_size: int = settings.chunking.chunk_size,
        chunk_overlap: int = settings.chunking.chunk_overlap,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step          = chunk_size - chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of documents. Returns flat list of all chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            logger.debug(f"Sliding window chunked document", extra={
                "source":      doc.metadata.get("source", "unknown"),
                "chunk_count": len(chunks),
            })
        logger.info(f"Sliding window chunking complete", extra={
            "total_chunks": len(all_chunks),
            "chunk_size":   self.chunk_size,
            "overlap":      self.chunk_overlap,
        })
        return all_chunks

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        tokens = split_into_tokens(doc.content)
        chunks = []
        chunk_index = 0

        i = 0
        while i < len(tokens):
            # Slice the token window
            window = tokens[i : i + self.chunk_size]
            text   = tokens_to_text(window)

            chunk = Chunk(
                content=text,
                metadata={
                    **doc.metadata,          # inherit all source metadata
                    "strategy":    "sliding_window",
                    "chunk_index": chunk_index,
                    "chunk_start": i,        # token position in original doc
                    "chunk_end":   i + len(window),
                }
            )
            chunks.append(chunk)
            chunk_index += 1
            i += self.step

        return chunks


# ─────────────────────────────────────────────
# 4. Strategy 2: Semantic Chunker
# ─────────────────────────────────────────────

class SemanticChunker:
    """
    Splits text at points where the meaning shifts significantly.

    Algorithm:
        1. Split document into sentences
        2. Embed each sentence using a sentence transformer
        3. Compute cosine distance between consecutive sentences
        4. Find sentences where distance > threshold (meaning shifted)
        5. Those sentences become chunk boundaries
        6. If a chunk exceeds max_chunk_size, split it further

    The threshold is the key hyperparameter:
        Low threshold  → splits at small meaning shifts → many small chunks
        High threshold → only splits at big shifts     → fewer large chunks

    Math — cosine distance between vectors A and B:
        distance = 1 - (A · B) / (||A|| × ||B||)

    Where:
        A · B   = dot product = sum of element-wise products
        ||A||   = L2 norm = sqrt(sum of squared elements)
    """

    def __init__(
        self,
        threshold: float  = 0.3,
        max_chunk_size: int = settings.chunking.chunk_size * 2,
        model_name: str   = settings.embedding.dense_model,
    ):
        """
        Args:
            threshold:      Cosine distance above which we place a boundary.
                            0.3 is a good starting point — we'll tune this.
            max_chunk_size: Safety cap — prevents runaway large chunks.
            model_name:     Sentence transformer model for embeddings.
        """
        self.threshold      = threshold
        self.max_chunk_size = max_chunk_size
        self.model_name     = model_name
        self._model         = None   # lazy load — don't load until needed

    @property
    def model(self):
        """
        Lazy loading: only load the embedding model when first needed.
        Loading a sentence transformer takes ~2 seconds and uses ~500MB RAM.
        We don't want that cost at import time.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model", extra={"model": self.model_name})
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.
        Handles: "End. Start" "End! Start" "End? Start"
        Avoids splitting on: "Dr. Smith" "e.g." "Fig. 3"
        """
        # Split on period/!/? followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Filter out empty or very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences

    def _cosine_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine distance between two embedding vectors.

        cosine_similarity = dot(A, B) / (norm(A) * norm(B))
        cosine_distance   = 1 - cosine_similarity

        Distance 0   = identical meaning
        Distance 1   = completely unrelated
        Distance 1.4 = maximum (opposite directions, rare)
        """
        dot_product  = np.dot(vec_a, vec_b)
        norm_a       = np.linalg.norm(vec_a)
        norm_b       = np.linalg.norm(vec_b)

        # Guard against zero vectors (empty or padding sentences)
        if norm_a == 0 or norm_b == 0:
            return 1.0

        similarity = dot_product / (norm_a * norm_b)
        # Clip to [-1, 1] to handle floating point errors
        similarity = np.clip(similarity, -1.0, 1.0)
        return float(1.0 - similarity)

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            logger.debug(f"Semantic chunked document", extra={
                "source":      doc.metadata.get("source", "unknown"),
                "chunk_count": len(chunks),
            })
        logger.info(f"Semantic chunking complete", extra={
            "total_chunks": len(all_chunks),
            "threshold":    self.threshold,
        })
        return all_chunks

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        sentences = self._split_into_sentences(doc.content)

        if len(sentences) == 0:
            return []

        if len(sentences) == 1:
            return [Chunk(
                content=sentences[0],
                metadata={**doc.metadata, "strategy": "semantic", "chunk_index": 0}
            )]

        # Embed all sentences at once (batching is much faster than one by one)
        logger.debug(f"Embedding {len(sentences)} sentences")
        embeddings = self.model.encode(sentences, batch_size=32, show_progress_bar=False)

        # Find chunk boundaries
        boundaries = [0]  # always start a chunk at sentence 0

        for i in range(len(sentences) - 1):
            distance = self._cosine_distance(embeddings[i], embeddings[i + 1])
            if distance > self.threshold:
                boundaries.append(i + 1)

        boundaries.append(len(sentences))  # end boundary

        # Build chunks from boundaries
        chunks = []
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end   = boundaries[idx + 1]
            text  = " ".join(sentences[start:end])

            # Safety: if chunk is too large, split it further with sliding window
            if len(text.split()) > self.max_chunk_size:
                sub_chunker = SlidingWindowChunker(
                    chunk_size=settings.chunking.chunk_size,
                    chunk_overlap=settings.chunking.chunk_overlap
                )
                sub_doc    = Document(content=text, metadata=doc.metadata)
                sub_chunks = sub_chunker._chunk_document(sub_doc)
                for sc in sub_chunks:
                    sc.metadata["strategy"] = "semantic"
                    sc.metadata["chunk_index"] = len(chunks)
                    chunks.extend([sc])
            else:
                chunks.append(Chunk(
                    content=text,
                    metadata={
                        **doc.metadata,
                        "strategy":       "semantic",
                        "chunk_index":    idx,
                        "sentence_start": start,
                        "sentence_end":   end,
                    }
                ))

        return chunks


# ─────────────────────────────────────────────
# 5. Strategy 3: Hierarchical Chunker
# ─────────────────────────────────────────────

class HierarchicalChunker:
    """
    Creates chunks at two levels simultaneously:
        Level 1 (parent): Large chunks — sections, ~1024 tokens
        Level 2 (child):  Small chunks — paragraphs, ~256 tokens

    Each child chunk stores the ID of its parent chunk.

    Why this matters for retrieval:
        Search finds the precise small child chunk.
        But you return the larger parent chunk to the LLM for more context.
        This is called "small-to-big retrieval" or "parent document retrieval".

    Structure:
        parent_0: [large section of text]
            child_0_0: [paragraph 1 of section]
            child_0_1: [paragraph 2 of section]
            child_0_2: [paragraph 3 of section]
        parent_1: [next large section]
            child_1_0: [paragraph 1]
            ...
    """

    def __init__(
        self,
        parent_chunk_size: int = settings.chunking.chunk_size * 2,
        child_chunk_size:  int = settings.chunking.chunk_size // 2,
        overlap:           int = settings.chunking.chunk_overlap,
    ):
        self.parent_chunker = SlidingWindowChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=overlap
        )
        self.child_chunker = SlidingWindowChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=overlap
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """
        Returns ALL chunks (parents + children) in a flat list.
        Use metadata["level"] to distinguish them:
            "parent" → large context chunks
            "child"  → small searchable chunks
        """
        all_chunks = []

        for doc in documents:
            parent_chunks = self.parent_chunker._chunk_document(doc)

            for parent_idx, parent_chunk in enumerate(parent_chunks):
                parent_id = f"{doc.metadata.get('source','doc')}__parent_{parent_idx}"

                # Tag the parent
                parent_chunk.metadata.update({
                    "strategy":  "hierarchical",
                    "level":     "parent",
                    "parent_id": parent_id,
                })
                all_chunks.append(parent_chunk)

                # Create children from this parent
                parent_as_doc = Document(
                    content=parent_chunk.content,
                    metadata=parent_chunk.metadata
                )
                child_chunks = self.child_chunker._chunk_document(parent_as_doc)

                for child_idx, child_chunk in enumerate(child_chunks):
                    child_chunk.metadata.update({
                        "strategy":  "hierarchical",
                        "level":     "child",
                        "parent_id": parent_id,
                        "child_index": child_idx,
                    })
                    all_chunks.append(child_chunk)

        parents  = sum(1 for c in all_chunks if c.metadata.get("level") == "parent")
        children = sum(1 for c in all_chunks if c.metadata.get("level") == "child")
        logger.info(f"Hierarchical chunking complete", extra={
            "parents": parents, "children": children
        })

        return all_chunks


# ─────────────────────────────────────────────
# 6. Chunker factory — select by name
# ─────────────────────────────────────────────

def get_chunker(strategy: str, **kwargs):
    """
    Factory function — get a chunker by strategy name.
    Used by the evaluation framework to run all strategies.

    Usage:
        chunker = get_chunker("sliding_window")
        chunker = get_chunker("semantic", threshold=0.25)
        chunker = get_chunker("hierarchical")
    """
    strategies = {
        "sliding_window": SlidingWindowChunker,
        "semantic":       SemanticChunker,
        "hierarchical":   HierarchicalChunker,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(strategies.keys())}"
        )

    return strategies[strategy](**kwargs)