import pandas as pd
import boto3
import streamlit as st
import json
import requests
import os
import base64
import mimetypes

# 1. El Entorno Virtual Aislado (venv_aws)

region = "us-east-1"
os.environ["AWS_REGION"] = region
llm_response = ""

# 2. El Repositorio de Código Git Vinculado (main branch en GitHub)


# 3. El Perfil de Autenticación Local de AWS CLI (profile_name="genaiday")
# Almacena de forma segura tus llaves de acceso (Access Key y Secret Key) en el sistema operativo Windows y las asocia a un perfil específico con 
# permisos controlados para interactuar con Amazon Bedrock.
# Creamos una "Sesión" de AWS.
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


#4. El Mapeador Multimodal (read_mime_type)
#Analiza dinámicamente cualquier extensión de archivo (incluyendo formatos web optimizados como .webp) 
# y extrae su identificador estándar de internet (MIME Type).

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


# =====================================================================
# 4. CAPA DE SERVICIO DE IA: INFERENCIA MULTIMODAL (VISIÓN + TEXTO)
# =====================================================================
# CORRECCIÓN DE NOMBRE: Renombramos a 'call_image' para coherencia con tu prueba final.

def call_image(file_path, caption, modelId="anthropic.claude-3-haiku-20240307-v1:0"):
    """
    Justificación: Orquesta la lectura del archivo del disco, su codificación binaria,
    y construye el payload mixto (Imagen + Instrucción) para el modelo visual.
    """
    
    # 1. LEER ARCHIVO Y CONVERTIR A BASE64 (Justificación Arquitectónica)
    # abrimos el archivo en modo 'rb' (read binary / lectura binaria).
    # .read() extrae los bytes crudos. base64.b64encode codifica esos bytes a formato Base64.
    # .decode('utf-8') convierte ese objeto de bytes codificado en un String de texto estándar.
    with open(file_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")

    # 2. ARMAR EL PAYLOAD MULTIMODAL
    # Fíjate que el array 'content' lleva DOS bloques: uno visual y uno de texto.
    config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    # Bloque 1: La imagen serializada
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            # Llamamos dinámicamente a nuestra función helper para el MIME exacto
                            "media_type": read_mime_type(file_path),
                            # Inyectamos el string larguísimo de la imagen codificada
                            "data": base64_string
                        }
                    },
                    # Bloque 2: La instrucción o pregunta sobre la imagen
                    {
                        "type": "text",
                        "text": caption
                    }
                ]
            }
        ]
    }

    # 3. INVOCACIÓN Y EXTRACCIÓN (Mismo patrón industrial del call_text)
    body = json.dumps(config)
    
    response = client.invoke_model(
        body=body, 
        modelId=modelId, 
        accept="application/json", 
        contentType="application/json"
    )
    
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")

"""
##################################################
# Validar conexion al modelo de AWS bedrock
#################################################
# 1. Procesamiento de Lenguaje Natural como input para el LLM
prompt = "Estoy por abrir una cafeteria al paso, recomiendame 5 nombres"
call_text(prompt, "anthropic.claude-3-haiku-20240307-v1:0")  # Si se quiere se puede cambiar el modelo (modificando el Model_id, que se tienen listado en AWS Bedrock)

# 2. Procesamiento de Imagen como input para el LLM
pic_path = "./meetup_test_image.jpg"
caption = "Cuantas personas hay en la imagen? cuantas laptos ves? cuantos usan gorro o sombrero?, de que color es el hoddie de la primer persona a la derecha de la foto?"
print(call_image(pic_path,caption,"anthropic.claude-3-haiku-20240307-v1:0"))
"""

# =====================================================================
# 5. BLOQUE DE VALIDACIÓN Y PRUEBAS DE CONECTIVIDAD
# =====================================================================

if __name__ == "__main__":
    print("--- PRUEBA 1: TEXTO SOLO (NLP) ---")
    prompt_nlp = "Estoy por abrir una cafeteria al paso, recomiendame 5 nombres"
    print(call_text(prompt_nlp))
    print("\n" + "="*50 + "\n")

    # NOTA PARA EJECUTAR LA PRUEBA 2: 
    # Asegúrate de tener una imagen real llamada 'meetup_test_image.jpg' en la misma carpeta,
    # de lo contrario Python lanzará un error FileNotFoundError al intentar abrir el archivo.
    # print("--- PRUEBA 2: MULTIMODAL (VISIÓN) ---")
    # pic_path = "./meetup_test_image.jpg"
    # caption_text = "Cuantas personas hay en la imagen? cuantas laptos ves?"
    # print(call_image(pic_path, caption_text))