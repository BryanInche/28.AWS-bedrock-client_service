import os

from .base import LLMProvider
from .bedrock_llm import BedrockProvider
from .groq_open_llm import GroqProvider
from .finetuned_llm import FinetunedProvider
from .cached_provider import CachedProvider

_PROVIDERS = {
    "bedrock": BedrockProvider,
    "groq": GroqProvider,
    "finetuned": FinetunedProvider,
}


def get_provider() -> LLMProvider:
    """
    LLM_PROVIDER: "bedrock", "groq" o "finetuned"
    USE_CACHE=true: envuelve el provider elegido con cache semántico
                     sobre Redis (requiere Redis Stack corriendo).
    """
    provider_name = os.getenv("LLM_PROVIDER", "groq").lower()

    provider_class = _PROVIDERS.get(provider_name)
    if provider_class is None:
        disponibles = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"LLM_PROVIDER='{provider_name}' no es válido. Opciones disponibles: {disponibles}"
        )

    provider = provider_class()

    if os.getenv("USE_CACHE", "false").lower() == "true":
        provider = CachedProvider(provider)

    return provider