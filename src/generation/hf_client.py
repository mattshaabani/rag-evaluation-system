"""
src/generation/hf_client.py

HuggingFace Inference API client.
Free alternative to running models locally via Ollama.
No downloads, no GPU needed — just an API token.

Free tier: ~1000 requests/day, plenty for development.

Usage:
    from src.generation.hf_client import HuggingFaceClient
    client   = HuggingFaceClient()
    response = client.generate(system="You are helpful.", user="What is RAG?")
    print(response.content)
"""

import time
import requests
from src.generation.ollama_client import LLMResponse
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceClient:
    """
    Client for HuggingFace Inference API.

    How it works:
        HuggingFace hosts models on their servers.
        We send text to their API, they run inference,
        they return the generated text.
        Free tier runs on shared infrastructure — sometimes
        a model is "loading" (cold start ~20s wait).

    Difference from Ollama:
        Ollama: model runs on YOUR machine (private, fast after load)
        HF API: model runs on HF servers (no download, rate limited)
    """

    def __init__(
        self,
        model:       str   = settings.generation.model,
        temperature: float = settings.generation.temperature,
        max_tokens:  int   = settings.generation.max_tokens,
        api_token:   str   = settings.env.hf_api_token,
        api_url:     str   = settings.env.hf_api_url,
    ):
        if not api_token:
            raise ValueError(
                "HuggingFace API token not set. "
                "Add HF_API_TOKEN=hf_... to your .env file."
            )

        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.api_url     = f"{api_url}/{model}"
        self.headers     = {"Authorization": f"Bearer {api_token}"}

        logger.info(f"Initialized HuggingFaceClient", extra={
            "model": model,
        })

    def _build_prompt(self, system: str, user: str) -> str:
        """
        Format system + user messages into a single prompt string.

        Most instruct models expect a specific chat template.
        Mistral format:
            <s>[INST] <<SYS>>
            {system}
            <</SYS>>
            {user} [/INST]

        This tells the model which part is instruction vs question.
        """
        return (
            f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
        )

    def health_check(self) -> bool:
        """Check if the API token is valid."""
        try:
            response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=self.headers,
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        system:      str,
        user:        str,
        temperature: float | None = None,
        max_tokens:  int   | None = None,
    ) -> LLMResponse:
        """
        Generate a response using HuggingFace Inference API.

        Handles cold starts: if the model is loading on HF servers,
        we wait and retry automatically (up to 3 times).
        """
        prompt = self._build_prompt(system, user)

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature":  temperature or self.temperature,
                "max_new_tokens": max_tokens or self.max_tokens,
                "return_full_text": False,  # only return new tokens
                "do_sample": True,
            }
        }

        start_time = time.time()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )

                # Model is loading on HF servers — wait and retry
                if response.status_code == 503:
                    wait_time = response.json().get("estimated_time", 20)
                    logger.info(f"Model loading on HF servers", extra={
                        "wait_seconds": round(wait_time),
                        "attempt": attempt + 1,
                    })
                    time.sleep(min(wait_time, 30))
                    continue

                response.raise_for_status()
                break

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise TimeoutError("HuggingFace API timed out after 3 attempts.")
                logger.warning(f"Request timed out, retrying", extra={"attempt": attempt + 1})
                time.sleep(5)

        latency_ms = (time.time() - start_time) * 1000
        data       = response.json()

        # HF returns a list of generated texts
        if isinstance(data, list):
            content = data[0].get("generated_text", "")
        else:
            content = data.get("generated_text", "")

        # Approximate token counts (HF free tier doesn't return exact counts)
        prompt_tokens = len(prompt.split())
        output_tokens = len(content.split())

        llm_response = LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

        logger.info(f"HF generation complete", extra={
            "model":      self.model,
            "latency_ms": round(latency_ms),
            "output_tokens": output_tokens,
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