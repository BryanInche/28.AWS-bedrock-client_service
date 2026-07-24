import os

from .base import LLMProvider
from .bedrock_llm import BedrockProvider
from .groq_open_llm import GroqProvider

_PROVIDERS = {
    "bedrock": BedrockProvider,
    "groq": GroqProvider,
}


def get_provider() -> LLMProvider:
    """
    Lee la variable de entorno LLM_PROVIDER ("bedrock" o "groq") y
    devuelve una instancia lista para usar.

    Esto es lo que te permite, sin tocar main.py ni ninguna otra parte
    del backend, cambiar de proveedor solo editando el .env:

        LLM_PROVIDER=groq      -> usa Llama 3 vía Groq (sin cuenta AWS)
        LLM_PROVIDER=bedrock   -> usa Claude vía AWS Bedrock

    Si mañana agregas OpenAI, solo creas openai_provider.py y lo
    registras en el diccionario _PROVIDERS de arriba.
    """
    provider_name = os.getenv("LLM_PROVIDER", "groq").lower()

    provider_class = _PROVIDERS.get(provider_name)
    if provider_class is None:
        disponibles = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"LLM_PROVIDER='{provider_name}' no es válido. Opciones disponibles: {disponibles}"
        )

    return provider_class()