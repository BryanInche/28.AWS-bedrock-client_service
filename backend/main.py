# main.py
import os

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from llm_providers import get_provider

load_dotenv()

# =====================================================================
# 1. INICIALIZACIÓN DE FASTAPI
# =====================================================================
app = FastAPI(
    title="API del Agente de Contenido",
    description="Backend modular para generación de contenido, con LLM intercambiable (Bedrock/Groq)",
    version="1.3.0",
)

# =====================================================================
# 2. PROVEEDOR DE LLM (se decide UNA vez, al arrancar el servidor,
#    según la variable de entorno LLM_PROVIDER)
# =====================================================================
llm = get_provider()

# =====================================================================
# 3. CONFIGURACIÓN DE CORS
# =====================================================================
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
    model_id: str | None = None  # si no se especifica, cada provider usa su default


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_SIZE_MB = 5


# =====================================================================
# 4. ENDPOINTS
# =====================================================================
@app.get("/")
def health_check():
    return {"status": "ok", "provider": os.getenv("LLM_PROVIDER", "groq")}


@app.post("/api/v1/content/generate-text")
def generate_text_endpoint(request: TextRequest):
    try:
        response_text = llm.call_text(prompt=request.prompt, model=request.model_id)
        return {"status": "success", "type": "text", "result": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor de IA: {str(e)}")


@app.post("/api/v1/content/analyze-image")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    caption: str = Form(...),
    model_id: str | None = Form(None),
):
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido: {file.content_type}")

        image_bytes = await file.read()
        if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"El archivo excede el límite de {MAX_IMAGE_SIZE_MB}MB.")

        response_analysis = llm.call_image(
            image_bytes=image_bytes,
            filename=file.filename,
            caption=caption,
            model=model_id,
        )
        return {"status": "success", "type": "multimodal", "result": response_analysis}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor multimodal: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

