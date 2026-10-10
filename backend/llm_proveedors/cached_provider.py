"""
Envuelve cualquier LLMProvider (Groq, Bedrock, Finetuned) con un cache
semántico sobre Redis. Mismo contrato LLMProvider — el resto del
backend ni se entera de que hay cache por debajo.

Patrón "Decorator": CachedProvider no ES un LLM, ENVUELVE a uno.
"""

from cache.semantic_cache import SemanticCache  # cache/ es hermana de llm_proveedors/, ambas dentro de backend/
from .base import LLMProvider


class CachedProvider(LLMProvider):
    def __init__(self, provider_envuelto: LLMProvider):
        self.provider = provider_envuelto
        self.cache = SemanticCache()

    def call_text(self, prompt: str, model: str | None = None) -> str:
        respuesta_cacheada = self.cache.get(prompt)
        if respuesta_cacheada is not None:
            return respuesta_cacheada

        respuesta = self.provider.call_text(prompt, model=model)
        self.cache.set(prompt, respuesta)
        return respuesta

    def call_image(self, image_bytes: bytes, filename: str, caption: str, model: str | None = None) -> str:
        # Las imágenes no se cachean en esta primera versión.
        return self.provider.call_image(image_bytes, filename, caption, model=model)