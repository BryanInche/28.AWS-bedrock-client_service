import base64
import mimetypes
import os

from groq import Groq

from .base import LLMProvider

# IMPORTANTE: verifica los nombres vigentes en console.groq.com/docs/models
# antes de usar en producción; Groq actualiza/retira modelos con frecuencia.
DEFAULT_TEXT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_VISION_MODEL = "llama-3.2-11b-vision-preview"


class GroqProvider(LLMProvider):
    """
    Proveedor que usa la API de Groq (modelos open source como Llama 3
    corriendo en su hardware LPU, muy baja latencia). Cumple el mismo
    contrato LLMProvider que BedrockProvider, así que main.py puede
    usar cualquiera de los dos sin cambiar su código.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Falta GROQ_API_KEY en las variables de entorno. "
                "Consíguela gratis en https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)

    def _read_mime_type(self, filename: str) -> str:
        mimetypes.add_type("image/webp", ".webp")
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type

    def call_text(self, prompt: str, model: str | None = None) -> str:
        model = model or DEFAULT_TEXT_MODEL

        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return completion.choices[0].message.content

    def call_image(self, image_bytes: bytes, filename: str, caption: str, model: str | None = None) -> str:
        model = model or DEFAULT_VISION_MODEL

        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = self._read_mime_type(filename)

        # Groq sigue el formato "OpenAI-compatible": la imagen va como
        # una URL de datos (data URI), no como un bloque separado tipo
        # Anthropic/Bedrock.
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_string}"},
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        return completion.choices[0].message.content