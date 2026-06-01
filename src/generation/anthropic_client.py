"""
src/generation/anthropic_client.py

Anthropic Claude API client.
Uses Claude as the LLM backend for our RAG pipeline.

Free tier is enough for the entire project.
Model: claude-haiku-4-5-20251001 — fastest and cheapest Claude model.

Usage:
    from src.generation.anthropic_client import AnthropicClient
    client   = AnthropicClient()
    response = client.generate(
        system="You are helpful.",
        user="What is RAG?",
    )
    print(response.content)
"""

import time
import anthropic
from src.generation.ollama_client import LLMResponse
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicClient:
    """
    Client for Anthropic Claude API.

    Why Claude for RAG?
        - Large context window (handles many retrieved chunks)
        - Follows instructions precisely (stays grounded in context)
        - Fast response times with Haiku model
        - Free tier sufficient for development and evaluation
    """

    def __init__(
        self,
        model:       str   = settings.generation.model,
        temperature: float = settings.generation.temperature,
        max_tokens:  int   = settings.generation.max_tokens,
        api_key:     str   = settings.env.anthropic_api_key,
    ):
        if not api_key:
            raise ValueError(
                "Anthropic API key not set. "
                "Add ANTHROPIC_API_KEY=sk-ant-... to your .env file."
            )

        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.client      = anthropic.Anthropic(api_key=api_key)

        logger.info(f"Initialized AnthropicClient", extra={
            "model": model,
        })

    def health_check(self) -> bool:
        """Verify the API key works by making a minimal request."""
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception as e:
            logger.warning(f"Health check failed", extra={"error": str(e)})
            return False

    def generate(
        self,
        system:      str,
        user:        str,
        temperature: float | None = None,
        max_tokens:  int   | None = None,
    ) -> LLMResponse:
        """
        Generate a response using Claude API.

        Args:
            system:      System message — sets Claude's role and rules.
            user:        User message — the actual question/prompt.
            temperature: Override instance temperature for this call.
            max_tokens:  Override instance max_tokens for this call.

        Returns:
            LLMResponse with content, token counts, and latency.
        """
        logger.debug(f"Sending request to Anthropic", extra={
            "model":      self.model,
            "user_len":   len(user),
            "system_len": len(system),
        })

        start_time = time.time()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            system=system,
            messages=[
                {"role": "user", "content": user}
            ],
        )

        latency_ms = (time.time() - start_time) * 1000

        content       = response.content[0].text
        prompt_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        llm_response = LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

        logger.info(f"Anthropic generation complete", extra={
            "model":          self.model,
            "total_tokens":   llm_response.total_tokens,
            "latency_ms":     round(latency_ms),
            "tokens_per_sec": round(llm_response.tokens_per_second, 1),
        })

        return llm_response

    def generate_from_template(
        self,
        template_output: dict,
        **kwargs,
    ) -> LLMResponse:
        """Same interface as OllamaClient for drop-in compatibility."""
        return self.generate(
            system=template_output["system"],
            user=template_output["user"],
            **kwargs,
        )