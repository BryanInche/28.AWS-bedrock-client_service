# /script.py
import base64
import json
import mimetypes
import os

import boto3  # Libreria para conectar con AWS
from dotenv import load_dotenv # Libreria para manejo de varibles de entorno

# =====================================================================
# 1. CARGA DE VARIABLES DE ENTORNO
# =====================================================================
# load_dotenv() busca un archivo ".env" en el directorio del proyecto
# y copia su contenido a os.environ. Es una función "silenciosa":
# - En LOCAL: si backend/.env existe, lo carga (AWS_ACCESS_KEY_ID, etc.)
# - En LA NUBE (ECS/Lambda): si no hay archivo .env, simplemente no
#   hace nada y no lanza error. Ahí las credenciales ya vienen del
#   IAM Role del servicio, no de este archivo.
# Por eso el mismo código sirve para ambos entornos sin cambiar nada.
load_dotenv()

# =====================================================================
# 2. CONFIGURACIÓN DE REGIÓN Y SESIÓN DE AWS
# =====================================================================
region = os.getenv("AWS_REGION", "us-east-1")

# boto3.session.Session sin "profile_name" sigue la cadena de
# resolución de credenciales por defecto:
#   1. Variables de entorno (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
#      -> vienen de tu .env en local, gracias a load_dotenv()
#   2. Rol IAM del servicio (ECS/Lambda/EC2) -> en producción en AWS
#   3. Archivo ~/.aws/credentials, perfil "default" -> fallback local
aws = boto3.session.Session(region_name=region)
client = aws.client("bedrock-runtime")

# Nota: en AWS hay dos clientes para Bedrock:
# - "bedrock": administración de modelos, fine-tuning, custom models.
# - "bedrock-runtime": inferencia en tiempo real (el que usamos aquí).


# =============================================================================
# 3. UTILIDAD: DETECCIÓN DE TIPO MIME (Identificar que tipo de archivo ingresa)
# =============================================================================
def read_mime_type(file_name):
    """
    Deduce el tipo MIME (ej. 'image/jpeg') a partir del nombre/extensión
    de un archivo. Bedrock exige este dato explícito en el campo
    'media_type' del payload multimodal; no lo infiere solo.
    """
    # Python no reconoce .webp por defecto en algunas versiones;
    # se registra manualmente antes de preguntar.
    mimetypes.add_type("image/webp", ".webp")
    mime_type, _ = mimetypes.guess_type(file_name)
    return mime_type


# =====================================================================
# 4. INFERENCIA DE TEXTO (LLM Solo procesa texto)
# =====================================================================
def call_text(prompt, modelId="anthropic.claude-3-haiku-20240307-v1:0"):
    """
    Envía un prompt de solo texto a un modelo Claude en Bedrock y
    devuelve el texto de la respuesta.
    """
    config = {
        "anthropic_version": "bedrock-2023-05-31",  # esquema de API de Anthropic
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    response = client.invoke_model(
        body=json.dumps(config),
        modelId=modelId,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


# =====================================================================
# 5. INFERENCIA MULTIMODAL (VISIÓN + TEXTO)
# =====================================================================
def call_image(image_bytes, filename, caption, modelId="anthropic.claude-3-haiku-20240307-v1:0"):
    """
    Envía una imagen (en bytes, ya leída por el endpoint de FastAPI)
    junto con una instrucción de texto, y devuelve el análisis del modelo.
    """
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
                            "media_type": read_mime_type(filename),
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": caption},
                ],
            }
        ],
    }

    response = client.invoke_model(
        body=json.dumps(config),
        modelId=modelId,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


# =====================================================================
# 6. BLOQUE DE VALIDACIÓN LOCAL
# =====================================================================
if __name__ == "__main__":
    print("--- PRUEBA 1: TEXTO SOLO (NLP) ---")
    prompt_nlp = "Estoy por abrir una cafeteria al paso, recomiendame 5 nombres"
    print(call_text(prompt_nlp))
