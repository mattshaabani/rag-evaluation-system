"""
src/generation/ollama_client.py

Ollama LLM client for local inference.
Ollama runs open-source models (Llama 3.1, Mistral, Phi-3)
completely locally — no API key, no cost, no data leaving your machine.

How Ollama works:
    1. You install Ollama (ollama.ai)
    2. You pull a model: `ollama pull llama3.1`
    3. Ollama runs a local HTTP server on port 11434
    4. We send requests to that server — same as any API

Usage:
    from src.generation.ollama_client import OllamaClient
    client   = OllamaClient()
    response = client.generate(
        system="You are helpful.",
        user="What is RAG?",
    )
    print(response.content)
"""

from readline import backend
import time
import requests
from dataclasses import dataclass
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 1. Response container
# ─────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Structured response from the LLM.
    Wraps raw API response with useful computed properties.
    """
    content:        str
    model:          str
    prompt_tokens:  int
    output_tokens:  int
    latency_ms:     float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms == 0:
            return 0.0
        return self.output_tokens / (self.latency_ms / 1000)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"LLMResponse("
            f"tokens={self.total_tokens}, "
            f"latency={self.latency_ms:.0f}ms, "
            f"preview='{preview}...')"
        )


# ─────────────────────────────────────────────
# 2. Ollama client
# ─────────────────────────────────────────────

class OllamaClient:
    """
    HTTP client for Ollama's local inference server.

    Ollama API endpoints we use:
        POST /api/chat    — multi-turn chat with system + user messages
        GET  /api/tags    — list available models
        GET  /           — health check

    Temperature explained:
        The LLM at each step computes a probability distribution
        over all possible next tokens. Temperature T scales the
        logits before softmax:

            p_i = exp(logit_i / T) / Σ exp(logit_j / T)

        T → 0:  distribution becomes a spike at the highest logit
                (always picks the most likely token, deterministic)
        T = 1:  original distribution unchanged
        T > 1:  distribution flattens (more random, more creative)

        For RAG we use T=0.1 — almost deterministic because we want
        factual, grounded answers, not creative ones.
    """

    def __init__(
        self,
        base_url:    str   = settings.env.ollama_base_url,
        model:       str   = settings.env.ollama_model,
        temperature: float = settings.generation.temperature,
        max_tokens:  int   = settings.generation.max_tokens,
        timeout:     int   = 120,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

        logger.info(f"Initialized OllamaClient", extra={
            "model":       model,
            "temperature": temperature,
            "base_url":    base_url,
        })

    def health_check(self) -> bool:
        """
        Check if Ollama server is running.
        Call this before the first generate() to give a clear
        error message if Ollama isn't started.
        """
        try:
            response = requests.get(
                f"{self.base_url}",
                timeout=5,
            )
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """Return list of models available in local Ollama."""
        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def generate(
        self,
        system:      str,
        user:        str,
        temperature: float | None = None,
        max_tokens:  int   | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            system:      System message — sets the LLM's role and rules.
            user:        User message — the actual question/prompt.
            temperature: Override instance temperature for this call.
            max_tokens:  Override instance max_tokens for this call.

        Returns:
            LLMResponse with content, token counts, and latency.

        Raises:
            ConnectionError: If Ollama server is not running.
            RuntimeError:    If the model returns an error.
        """
        # Check server is running
        if not self.health_check():
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: `ollama serve`"
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens  or self.max_tokens,
            },
        }

        logger.debug(f"Sending request to Ollama", extra={
            "model":      self.model,
            "user_len":   len(user),
            "system_len": len(system),
        })

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s. "
                f"Try a smaller model or increase timeout."
            )

        latency_ms = (time.time() - start_time) * 1000
        data       = response.json()

        # Extract content from Ollama response structure
        content = data.get("message", {}).get("content", "")

        # Token counts (Ollama provides these)
        prompt_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        llm_response = LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

        logger.info(f"LLM generation complete", extra={
            "model":         self.model,
            "total_tokens":  llm_response.total_tokens,
            "latency_ms":    round(latency_ms),
            "tokens_per_sec": round(llm_response.tokens_per_second, 1),
        })

        return llm_response

    def generate_from_template(
        self,
        template_output: dict,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate using the output of a PromptTemplate.format() call.
        Convenience method so you don't unpack the dict manually.

        Usage:
            template = RAGPromptTemplate()
            prompt   = template.format(question=q, chunks=chunks)
            response = client.generate_from_template(prompt)
        """
        return self.generate(
            system=template_output["system"],
            user=template_output["user"],
            **kwargs,
        )
    
    def get_llm_client():
        """
        Factory function — returns the right LLM client based on config.
        Every other file imports this instead of a specific client.

        Usage:
            from src.generation.ollama_client import get_llm_client
            client = get_llm_client()
            response = client.generate(system=..., user=...)
        """
        backend = settings.generation.backend

        if backend == "huggingface":
            from src.generation.hf_client import HuggingFaceClient
            return HuggingFaceClient()
        elif backend == "ollama":
            return OllamaClient()
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose 'ollama' or 'huggingface' in rag_config.yaml"
            )