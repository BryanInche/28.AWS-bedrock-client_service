# main.py
import os

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from script import call_text, call_image

# =====================================================================
# 1. INICIALIZACIÓN DE FASTAPI
# =====================================================================
app = FastAPI(
    title="API del Agente de Contenido",
    description="Backend modular para generación de contenido usando AWS Bedrock",
    version="1.2.0",
)

# =====================================================================
# 2. CONFIGURACIÓN DE CORS, para darle acceso a otras APIs al Backend
# =====================================================================
# CAMBIO IMPORTANTE:
# Antes la lista de orígenes estaba fija en el código (solo servía en
# local). Ahora se lee de la variable de entorno ALLOWED_ORIGINS, que
# en local puede venir de un .env y en producción se define en el
# servicio de despliegue (ECS task definition, Lambda env vars, etc.)
# Formato esperado: "https://miapp.com,https://www.miapp.com"
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    prompt: str
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"


# =====================================================================
# 3. ENDPOINTS
# =====================================================================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend modular en línea."}


@app.post("/api/v1/content/generate-text")
def generate_text_endpoint(request: TextRequest):
    try:
        response_text = call_text(prompt=request.prompt, modelId=request.model_id)
        return {"status": "success", "type": "text", "result": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor de IA: {str(e)}")


# CAMBIO IMPORTANTE (seguridad):
# Antes este endpoint recibía "file_path" como texto plano en el JSON y
# el servidor abría directamente esa ruta con open(request.file_path).
# Esto es una vulnerabilidad de "path traversal": cualquier cliente
# podía mandar una ruta arbitraria del sistema de archivos del
# servidor (ej. "../../.env" o una ruta absoluta) y el backend la leía
# sin más validación.
#
# Ahora el endpoint recibe el archivo real como "multipart/form-data"
# usando UploadFile de FastAPI. El backend nunca vuelve a tocar el
# disco del servidor con una ruta que no controla; simplemente lee los
# bytes que el cliente subió, y opcionalmente valida tipo/tamaño antes
# de mandarlos a Bedrock.
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_SIZE_MB = 5


@app.post("/api/v1/content/analyze-image")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    caption: str = Form(...),
    model_id: str = Form("anthropic.claude-3-haiku-20240307-v1:0"),
):
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido: {file.content_type}",
            )

        image_bytes = await file.read()

        if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo excede el límite de {MAX_IMAGE_SIZE_MB}MB.",
            )

        response_analysis = call_image(
            image_bytes=image_bytes,
            filename=file.filename,
            caption=caption,
            modelId=model_id,
        )

        return {"status": "success", "type": "multimodal", "result": response_analysis}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor multimodal: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

