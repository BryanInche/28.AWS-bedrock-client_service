# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# =====================================================================
# 1. IMPORTACIÓN DE NUESTRA CAPA DE IA (Tus funciones de script.py)
# =====================================================================
# Aquí ocurre la magia. Python busca el archivo 'script.py' en la misma carpeta
# e importa las funciones específicas que ya programaste y optimizaste.
from script import call_text, call_image

# =====================================================================
# 2. INICIALIZACIÓN DE FASTAPI Y ESQUEMAS DE VALIDACIÓN
# =====================================================================
app = FastAPI(
    title="API del Agente de Contenido",
    description="Backend modular para generación de contenido usando AWS Bedrock",
    version="1.1.0"
)

# Definimos la estructura limpia de los datos que nuestra API va a recibir
class TextRequest(BaseModel):
    prompt: str
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"

class ImageRequest(BaseModel):
    file_path: str
    caption: str
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"

# =====================================================================
# 3. ENDPOINTS DE LA API (Rutas que llamará Vue.js)
# =====================================================================

@app.get("/")
def health_check():
    """Ruta para verificar que el servidor de FastAPI está encendido."""
    return {"status": "ok", "message": "Backend modular en línea."}


@app.post("/api/v1/content/generate-text")
def generate_text_endpoint(request: TextRequest):
    """
    Recibe el prompt del usuario en formato JSON, llama a la función
    'call_text' de script.py y retorna la respuesta procesada.
    """
    try:
        # LLAMADA DIRECTA A TU FUNCIÓN DE SCRIPT.PY
        response_text = call_text(prompt=request.prompt, modelId=request.model_id)
        
        return {
            "status": "success",
            "type": "text",
            "result": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor de IA: {str(e)}")


@app.post("/api/v1/content/analyze-image")
def analyze_image_endpoint(request: ImageRequest):
    """
    Recibe la ruta de una imagen y una instrucción, llama a la función
    'call_image' de script.py y retorna el análisis visual del modelo.
    """
    try:
        # Validamos primero que el archivo físico realmente exista en el backend
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="El archivo de imagen no fue encontrado en el servidor.")

        # LLAMADA DIRECTA A TU FUNCIÓN MULTIMODAL DE SCRIPT.PY
        response_analysis = call_image(
            file_path=request.file_path, 
            caption=request.caption, 
            modelId=request.model_id
        )

        return {
            "status": "success",
            "type": "multimodal",
            "result": response_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor multimodal: {str(e)}")
    

# uvicorn main:app --reload  # reload, para reiniciar el server cada que cambias algo en el code
# Encender el servidor desde main.py:
if __name__ == "__main__":
    import uvicorn
    # Le decimos a Python que él mismo encienda el servidor Uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)