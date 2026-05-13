"""LLM factory: creates the correct chat model based on provider."""
import os
from typing import Optional

from langchain_openai import ChatOpenAI


def create_llm(model_name: str, temperature: float = 0.1, max_tokens: int = 4096):
    provider = model_name.split("/")[0] if "/" in model_name else model_name
    actual_model = model_name.split("/", 1)[1] if "/" in model_name else model_name

    endpoint_map = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
        },
        "openai": {
            "base_url": None,
            "api_key_env": "OPENAI_API_KEY",
        },
        "google": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key_env": "GEMINI_API_KEY",
        },
        "meta-llama": {
            "base_url": None,
            "api_key_env": None,
        },
        "anthropic": {
            "base_url": None,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
    }

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            pass

    endpoint = endpoint_map.get(provider, endpoint_map["openai"])
    api_key = None
    if endpoint.get("api_key_env"):
        api_key = os.environ.get(endpoint["api_key_env"])

    kwargs = {
        "model": actual_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider not in ("openai", "anthropic"):
        kwargs["openai_api_key"] = api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")

    if endpoint.get("base_url"):
        kwargs["openai_api_base"] = endpoint["base_url"]

    return ChatOpenAI(**kwargs)
