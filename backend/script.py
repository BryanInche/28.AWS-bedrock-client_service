# script.py

# En LOCAL (tu laptop, desarrollo)
from dotenv import load_dotenv
load_dotenv()  # esto lee backend/.env y lo mete en os.environ

# En LA NUBE (ECS, Lambda, EC2)
# Permite que el mismo código sirva para los dos mundos.
aws = boto3.session.Session(region_name=region)


import pandas as pd
import boto3
import streamlit as st
import json
import requests
import os
import base64
import mimetypes

# =====================================================================
# 1. CONFIGURACIÓN DE REGIÓN Y SESIÓN DE AWS
# =====================================================================
region = "us-east-1"
os.environ["AWS_REGION"] = region
llm_response = ""

# 2. El Repositorio de Código Git Vinculado (main branch en GitHub)
# Antes: boto3.session.Session(profile_name="genaiday", region_name=region)
#
# El parámetro profile_name solo funciona en TU máquina, porque lee
# el archivo local ~/.aws/credentials que creaste con
# "aws configure --profile genaiday". Ese archivo NO existe dentro de
# un contenedor Docker ni en un runner de GitHub Actions.
#
# Al quitar profile_name, boto3 sigue su "cadena de resolución de
# credenciales" por defecto, en este orden:
#   1. Variables de entorno (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
#   2. Rol IAM asignado al contenedor/instancia (ECS, EC2, Lambda)
#   3. Archivo ~/.aws/credentials (perfil "default"), si existe
#
# En LOCAL: define las variables en un archivo .env (ya está en tu
# .gitignore) y cárgalas con python-dotenv, o usa
# "aws configure" (perfil default) sin profile_name.
#
# En PRODUCCIÓN (ECS/Lambda): usa un Role de IAM asociado al servicio,
# NUNCA pongas llaves de acceso como variables de entorno en producción
# si puedes evitarlo; el Role es más seguro.
# -----------------------------------------------------------------
aws = boto3.session.Session(region_name=region)
client = aws.client("bedrock-runtime")
 
#Recuerda que en AWS hay dos clientes para Bedrock: bedrock (para 
# administrar modelos, crear custom models y fine-tuning) y 
# bedrock-runtime (diseñado exclusivamente para la inferencia de baja 
# latencia en tiempo real).


# =====================================================================
# 2. UTILIDAD: DETECCIÓN DE TIPO MIME
# =====================================================================
# Función de Lectura de Archivos
# Esto dejará tu backend listo para cuando el agente reciba, 
# por ejemplo, adjuntos de reclamos en formato de imagen.
def read_mime_type(file_path):
    # # 1. El "hack": Por defecto, la librería de Python a veces no reconoce
    # el formato moderno '.webp' (muy usado en web para comprimir imágenes sin perder calidad).
    # Esta línea fuerza a Python a registrar que si ve un '.webp', su tipo MIME es 'image/webp'
    mimetypes.add_type("image/webp",".webp")

    # 2. 'guess_type' analiza el nombre del archivo (ej. "reclamo_cliente.jpeg")
    # y devuelve una tupla con dos valores: (tipo_mime, codificación).
    # Ejemplo: Para "foto.jpg", devuelve ('image/jpeg', None).
    mime_type = mimetypes.guess_type(file_path)


    # 3. Extraemos solo el primer elemento de la tupla (posición [0]), 
    # que es la cadena de texto con el tipo (ej. "image/jpeg") y lo retornamos.
    return mime_type[0]

# =====================================================================
# 3. INFERENCIA DE TEXTO
# =====================================================================
#Inference Payload nativo que requiere la API de Anthropic dentro de 
# Bedrock. anthropic_version es un parámetro obligatorio que exige AWS 
# para saber qué esquema de API de Claude estás invocando.
def call_text(prompt, modelId="anthropic.claude-3-haiku-20240307-v1:0"):
    config = {
            "anthropic_version" : "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content":[{
                        "type": "text",
                        "text": prompt
                    }]   
                }
            ]
    }

    body = json.dumps(config)
    modelId = modelId
    accept = "application/json"
    contentType = "application/json"


    #Realiza la llamada HTTP POST por debajo hacia los endpoints 
    # administrados de AWS. Pasas el payload serializado (body), el 
    # identificador único del modelo (modelId), y los headers HTTP 
    # estándar (accept y contentType) para informarle a AWS que tanto el 
    # envío como la recepción son JSON puros.
    response = client.invoke_model(
        body = body, modelId=modelId, accept=accept, 
        contentType=contentType)
    
    # Se usa json.loads() para deserializarlos de vuelta a un diccionario
    #  de Python, y finalmente navegas por las llaves del JSON de Anthropic 
    # (content -> posición 0 -> text) para aislar la respuesta en texto 
    # limpio del LLM.
    response_body= json.loads(response.get("body").read())
    results = response_body.get("content")[0].get("text")
    
    return results

# =====================================================================
# 4. INFERENCIA MULTIMODAL (VISIÓN + TEXTO)
# =====================================================================
# CAMBIO IMPORTANTE:
# Antes esta función recibía un file_path y lo abría directamente.
# Ahora recibe los BYTES crudos del archivo (image_bytes) y el nombre
# original (filename) solo para deducir el MIME type. Así el backend
# nunca vuelve a tocar el disco ni confía en una ruta enviada por el
# cliente. Ver el nuevo endpoint en main.py para el detalle de por qué.
def call_image(image_bytes, filename, caption, modelId="anthropic.claude-3-haiku-20240307-v1:0"):
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
 
    body = json.dumps(config)
 
    response = client.invoke_model(
        body=body,
        modelId=modelId,
        accept="application/json",
        contentType="application/json",
    )
 
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")

# =====================================================================
# 5. BLOQUE DE VALIDACIÓN Y PRUEBAS DE CONECTIVIDAD
# =====================================================================

if __name__ == "__main__":
    print("--- PRUEBA 1: TEXTO SOLO (NLP) ---")
    prompt_nlp = "Estoy por abrir una cafeteria al paso, recomiendame 5 nombres"
    print(call_text(prompt_nlp))

