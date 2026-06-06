"""
src/generation/prompt_templates.py

Prompt templates for the RAG pipeline.
Centralizes all prompts so they can be versioned, tested, and
swapped without touching any other code.

Three template types:
    RAGPromptTemplate     — answer questions from retrieved context
    EvaluationPromptTemplate — generate Q&A pairs for evaluation dataset
    QueryRewriteTemplate  — rewrite queries for better retrieval

Usage:
    from src.generation.prompt_templates import RAGPromptTemplate
    template = RAGPromptTemplate()
    prompt   = template.format(question="What is attention?", chunks=chunks)
"""

from dataclasses import dataclass
from src.data.chunker import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 1. Base template class
# ─────────────────────────────────────────────

@dataclass
class BasePromptTemplate:
    """
    Base class for all prompt templates.
    Each template has a name and version for tracking in MLflow.
    """
    name:    str = "base"
    version: str = "v1"

    def format(self, **kwargs) -> dict:
        """
        Format the template with provided variables.
        Returns a dict with 'system' and 'user' keys —
        matching the message format most LLMs expect.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────
# 2. RAG answer prompt
# ─────────────────────────────────────────────

class RAGPromptTemplate(BasePromptTemplate):
    """
    The core RAG prompt — answers questions from retrieved context.

    Design decisions:
        1. Explicit grounding instruction — "ONLY use the context below"
           This is the single most important line for faithfulness.

        2. Graceful degradation — "If the answer is not in the context,
           say 'I don't have enough information'". Without this the LLM
           hallucinates confidently.

        3. Source citation — ask the LLM to cite which source it used.
           Makes answers verifiable and trustworthy.

        4. Numbered context chunks — makes citation easier and helps
           the LLM reason about multiple sources.
    """
    name:    str = "rag_answer"
    version: str = "v1"

    SYSTEM_PROMPT = """You are a precise question-answering assistant.
Your job is to answer questions using ONLY the context provided below.

Rules you must follow:
1. ONLY use information from the provided context. Never use prior knowledge.
2. If the answer is not in the context, respond with exactly:
   "I don't have enough information in the provided context to answer this question."
3. Always cite the source of your answer using [Source: X] notation.
4. Be concise and direct. Do not repeat the question.
5. If multiple context chunks are relevant, synthesize them into one coherent answer."""

    USER_TEMPLATE = """CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:"""

    def _format_context(self, chunks: list[Chunk]) -> str:
        """
        Format retrieved chunks into a numbered context block.

        Example output:
            [1] Source: paper.pdf (page 3)
            Attention is a mechanism that allows the model to focus
            on relevant parts of the input...

            [2] Source: notes.txt
            The transformer architecture uses multi-head attention...
        """
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            # Extract source info from metadata
            source    = chunk.metadata.get("source", "unknown")
            page      = chunk.metadata.get("page", "")
            page_info = f" (page {page})" if page else ""

            lines.append(f"[{i}] Source: {source}{page_info}")
            lines.append(chunk.content.strip())
            lines.append("")   # blank line between chunks

        return "\n".join(lines)

    def format(self, question: str, chunks: list[Chunk]) -> dict:
        """
        Format the RAG prompt.

        Args:
            question: The user's question.
            chunks:   Retrieved context chunks.

        Returns:
            {"system": str, "user": str}
            Ready to send to any LLM API.
        """
        context_block = self._format_context(chunks)

        user_message = self.USER_TEMPLATE.format(
            context_block=context_block,
            question=question,
        )

        logger.debug(f"Formatted RAG prompt", extra={
            "template":     self.name,
            "version":      self.version,
            "n_chunks":     len(chunks),
            "question_len": len(question),
        })

        return {
            "system": self.SYSTEM_PROMPT,
            "user":   user_message,
        }


# ─────────────────────────────────────────────
# 3. Evaluation dataset generation prompt
# ─────────────────────────────────────────────

class EvaluationPromptTemplate(BasePromptTemplate):
    """
    Generates question-answer pairs from chunks for building
    the evaluation dataset.

    Why do we need this?
        To evaluate our RAG system we need ground truth Q&A pairs.
        We use an LLM to generate realistic questions from our
        document chunks — then manually review a sample.

    This is called "synthetic evaluation dataset generation" —
    a standard technique when you don't have human-labeled data.
    """
    name:    str = "eval_dataset_gen"
    version: str = "v1"

    SYSTEM_PROMPT = """You are an expert at creating evaluation datasets
for question-answering systems.

Given a text passage, generate realistic questions that:
1. Can be answered DIRECTLY from the passage
2. Test different aspects: facts, concepts, relationships
3. Vary in complexity: simple lookup to multi-step reasoning
4. Are specific enough that only this passage answers them well

Return ONLY valid JSON. No preamble, no explanation."""

    USER_TEMPLATE = """TEXT PASSAGE:
{chunk_content}

SOURCE: {source}

Generate {n_questions} question-answer pairs from this passage.

Return this exact JSON structure:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "...",
      "source": "{source}",
      "chunk_content": "{chunk_preview}"
    }}
  ]
}}"""

    def format(
        self,
        chunk: Chunk,
        n_questions: int = 3,
    ) -> dict:
        source        = chunk.metadata.get("source", "unknown")
        chunk_preview = chunk.content[:100].replace('"', "'")

        user_message = self.USER_TEMPLATE.format(
            chunk_content=chunk.content,
            source=source,
            n_questions=n_questions,
            chunk_preview=chunk_preview,
        )

        return {
            "system": self.SYSTEM_PROMPT,
            "user":   user_message,
        }


# ─────────────────────────────────────────────
# 4. Query rewriting prompt
# ─────────────────────────────────────────────

class QueryRewriteTemplate(BasePromptTemplate):
    """
    Rewrites a user query to improve retrieval quality.

    Why rewrite queries?
        Users ask questions conversationally:
        "What did they say about that attention thing?"

        But vector search works better with explicit terms:
        "attention mechanism transformer self-attention"

        Query rewriting bridges this gap.

    This technique is called HyDE (Hypothetical Document Embeddings)
    when taken further — generate a hypothetical answer, embed that
    instead of the question. We implement the simpler version here.
    """
    name:    str = "query_rewrite"
    version: str = "v1"

    SYSTEM_PROMPT = """You are an expert at reformulating search queries
to improve document retrieval.

Rewrite the given question into a clear, keyword-rich search query that:
1. Preserves the original intent completely
2. Expands abbreviations and acronyms
3. Adds relevant technical terms the answer would contain
4. Removes conversational filler words

Return ONLY the rewritten query. Nothing else."""

    USER_TEMPLATE = """Original question: {question}

Rewritten search query:"""

    def format(self, question: str) -> dict:
        return {
            "system": self.SYSTEM_PROMPT,
            "user":   self.USER_TEMPLATE.format(question=question),
        }


# ─────────────────────────────────────────────
# 5. Template registry
# ─────────────────────────────────────────────

TEMPLATES = {
    "rag_answer":    RAGPromptTemplate,
    "eval_gen":      EvaluationPromptTemplate,
    "query_rewrite": QueryRewriteTemplate,
}

def get_template(name: str, **kwargs) -> BasePromptTemplate:
    """
    Get a template by name.

    Usage:
        template = get_template("rag_answer")
        template = get_template("query_rewrite")
    """
    if name not in TEMPLATES:
        raise ValueError(
            f"Unknown template '{name}'. "
            f"Available: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[name](**kwargs)