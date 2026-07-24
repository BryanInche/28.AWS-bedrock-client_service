from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Contrato que debe cumplir CUALQUIER proveedor de LLM que conectemos
    (Bedrock, Groq, OpenAI, Gemini, etc.).

    ¿Por qué una clase abstracta y no simplemente funciones sueltas?
    Porque así garantizamos que todos los proveedores expongan
    exactamente la misma "forma" (mismos métodos, mismos parámetros de
    entrada/salida). El resto del backend (main.py) programa contra
    esta interfaz, no contra un proveedor específico — así podemos
    cambiar de proveedor sin tocar ni una línea de main.py.
    Esto se conoce como "Strategy Pattern".
    """

    @abstractmethod
    def call_text(self, prompt: str, model: str | None = None) -> str:
        """Envía un prompt de solo texto y devuelve la respuesta en texto plano."""
        raise NotImplementedError

    @abstractmethod
    def call_image(self, image_bytes: bytes, filename: str, caption: str, model: str | None = None) -> str:
        """Envía una imagen + instrucción de texto y devuelve el análisis en texto plano."""
        raise NotImplementedError