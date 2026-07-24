# /bedrock_llm.py
import base64
import json
import mimetypes
import os

import boto3

from .base import LLMProvider

DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


class BedrockProvider(LLMProvider):
    """
    Misma lógica que ya tenías en script.py, ahora encapsulada como
    una clase que cumple el contrato LLMProvider. El comportamiento
    interno NO cambió, solo su ubicación y forma (funciones -> métodos).
    """

    #  CONFIGURACIÓN DE REGIÓN Y SESIÓN DE AWS
    def __init__(self):
        region = os.getenv("AWS_REGION", "us-east-1")
        session = boto3.session.Session(region_name=region) # Alineado para Local y Cloud
        self.client = session.client("bedrock-runtime")

    # UTILIDAD: DETECCIÓN DE TIPO MIME (Identificar que tipo de archivo ingresa/ Imagenes)
    def _read_mime_type(self, filename: str) -> str:
        mimetypes.add_type("image/webp", ".webp")
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type

    # INFERENCIA DE TEXTO (LLM Solo procesa texto)
    def call_text(self, prompt: str, model: str | None = None) -> str:
        model = model or DEFAULT_MODEL
        config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }

        response = self.client.invoke_model(
            body=json.dumps(config),
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("content")[0].get("text")

    # INFERENCIA MULTIMODAL (VISIÓN + TEXTO)
    def call_image(self, image_bytes: bytes, filename: str, caption: str, model: str | None = None) -> str:
        model = model or DEFAULT_MODEL
        base64_string = base64.b64encode(image_bytes).decode("utf-8")

        config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": self._read_mime_type(filename),
                                "data": base64_string,
                            },
                        },
                        {"type": "text", "text": caption},
                    ],
                }
            ],
        }

        response = self.client.invoke_model(
            body=json.dumps(config),
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("content")[0].get("text")