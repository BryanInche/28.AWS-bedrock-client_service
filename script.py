import pandas as pd
import boto3
import streamlit as st
import json
import requests
import os
import base64
import mimetypes

region = "us-east-1"
os.environ["AWS_REGION"] = region
llm_response = ""

# 1. Creamos una "Sesión" de AWS.
# Esto lee el archivo oculto en tu computadora donde guardamos tus llaves secretas
# al ejecutar 'aws configure --profile genaiday'. Así, AWS sabe que eres tú
# y que tienes permisos, sin tener que escribir contraseñas en el código.
aws = boto3.session.Session(profile_name="genaiday", region_name=region) # Credenciales de AWS Bedrock

# 2. A partir de esa sesión segura, instanciamos el cliente específico para interactuar
# con los modelos de IA en tiempo real. 'bedrock-runtime' es el servicio diseñado
# exclusivamente para enviar prompts y recibir respuestas (inferencia).
client = aws.client("bedrock-runtime")  # Uso de cliente bedrock-runtime

#Recuerda que en AWS hay dos clientes para Bedrock: bedrock (para 
# administrar modelos, crear custom models y fine-tuning) y 
# bedrock-runtime (diseñado exclusivamente para la inferencia de baja 
# latencia en tiempo real).

##################################################
#Preparación Multimodal
####################################################
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

######################################################
#Función de Inferencia (call_text)
#######################################################
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
        body = body, modelId=modelId, accept=accept, contentType=contentType)
    
    # Se usa json.loads() para deserializarlos de vuelta a un diccionario
    #  de Python, y finalmente navegas por las llaves del JSON de Anthropic 
    # (content -> posición 0 -> text) para aislar la respuesta en texto 
    # limpio del LLM.
    response_body= json.loads(response.get("body").read())
    results = response_body.get("content")[0].get("text")
    
    return results



prompt = "Estoy por abrir una cafeteria al paso, recomiendame 5 nombres"
call_text(prompt, "anthropic.claude-3-haiku-20240307-v1:0")





