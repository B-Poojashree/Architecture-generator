"""
Module 6 - LLM Provider Module
================================
A single unified interface for calling six different LLM providers:
    - Gemini 2.5 Flash
    - DeepSeek V3
    - Groq (Llama 3.3 served via Groq)
    - Llama 3.3
    - Qwen 2.5
    - Mistral Small

Each provider has its own function so the Benchmark Agent (Module 10)
can call them independently and time each call. API keys are loaded
from a .env file - never hardcoded.
"""

import os
import time
import logging
from typing import Dict, Callable

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# API keys - loaded once from environment / .env
# ----------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


class LLMResponse:
    """Standardized response object returned by every provider function."""

    def __init__(self, model_name: str, text: str, latency_seconds: float, success: bool, error: str = ""):
        self.model_name = model_name
        self.text = text
        self.latency_seconds = latency_seconds
        self.success = success
        self.error = error

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "text": self.text,
            "latency_seconds": round(self.latency_seconds, 3),
            "success": self.success,
            "error": self.error,
        }


def _timed_call(model_name: str, call_fn: Callable[[], str]) -> LLMResponse:
    """
    Wrap any provider call with timing + uniform error handling so the
    Benchmark Agent can compare response_time across all five models
    on equal footing.
    """
    start = time.time()
    try:
        text = call_fn()
        elapsed = time.time() - start
        return LLMResponse(model_name=model_name, text=text, latency_seconds=elapsed, success=True)
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("[%s] call failed: %s", model_name, exc)
        return LLMResponse(model_name=model_name, text="", latency_seconds=elapsed, success=False, error=str(exc))


# ----------------------------------------------------------------------
# Provider 1: Gemini 2.5 Flash
# ----------------------------------------------------------------------
def call_gemini(prompt: str, temperature: float = 0.3) -> LLMResponse:
    """Call Google's Gemini 2.5 Flash model."""

    def _run():
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    return _timed_call("Gemini 2.5 Flash", _run)


# ----------------------------------------------------------------------
# Provider 2: DeepSeek V3
# ----------------------------------------------------------------------
def call_deepseek(prompt: str, temperature: float = 0.3) -> LLMResponse:
    """Call DeepSeek V3 via DeepSeek's OpenAI-compatible chat API."""

    def _run():
        if not DEEPSEEK_API_KEY:
            raise EnvironmentError("DEEPSEEK_API_KEY not set in .env")

        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return _timed_call("DeepSeek V3", _run)


# ----------------------------------------------------------------------
# Provider 3: Groq (fast inference host)
# ----------------------------------------------------------------------
def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.3) -> LLMResponse:
    """Call a model hosted on Groq's low-latency inference API."""

    def _run():
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set in .env")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return _timed_call(f"Groq ({model})", _run)


# ----------------------------------------------------------------------
# Provider 4: Llama 3.3 (routed through Groq by default)
# ----------------------------------------------------------------------
def call_llama(prompt: str, temperature: float = 0.3) -> LLMResponse:
    """Call Llama 3.3. Uses Groq as the hosting backend for low latency."""
    response = call_groq(prompt, model="llama-3.3-70b-versatile", temperature=temperature)
    response.model_name = "Llama 3.3"
    return response


# ----------------------------------------------------------------------
# Provider 5: Qwen 2.5
# ----------------------------------------------------------------------
def call_qwen(prompt: str, temperature: float = 0.3) -> LLMResponse:
    """Call Qwen 2.5 via Alibaba Cloud DashScope's OpenAI-compatible API."""

    def _run():
        if not QWEN_API_KEY:
            raise EnvironmentError("QWEN_API_KEY not set in .env")

        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {"Authorization": f"Bearer {QWEN_API_KEY}"}
        payload = {
            "model": "qwen2.5-72b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return _timed_call("Qwen 2.5", _run)


# ----------------------------------------------------------------------
# Provider 6: Mistral Small
# ----------------------------------------------------------------------
def call_mistral(prompt: str, temperature: float = 0.3) -> LLMResponse:
    """Call Mistral Small via Mistral AI's chat completions API."""

    def _run():
        if not MISTRAL_API_KEY:
            raise EnvironmentError("MISTRAL_API_KEY not set in .env")

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
        payload = {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return _timed_call("Mistral Small", _run)


# ----------------------------------------------------------------------
# Registry used by the Benchmark Agent to iterate all models
# ----------------------------------------------------------------------
LLM_PROVIDERS: Dict[str, Callable[[str], LLMResponse]] = {
    "gemini": call_gemini,
    "deepseek": call_deepseek,
    "llama": call_llama,
    "qwen": call_qwen,
    "mistral": call_mistral,
}


def call_all_providers(prompt: str) -> Dict[str, Dict]:
    """
    Call every registered LLM provider with the same prompt.
    Used by the Benchmark Agent to run identical prompts across models.
    """
    results = {}
    for key, fn in LLM_PROVIDERS.items():
        logger.info("Calling provider: %s", key)
        response = fn(prompt)
        results[key] = response.to_dict()
    return results


if __name__ == "__main__":
    sample_prompt = "List three functional requirements for a to-do list app."
    for name, response_dict in call_all_providers(sample_prompt).items():
        logger.info("%s -> success=%s, latency=%ss", name, response_dict["success"], response_dict["latency_seconds"])
