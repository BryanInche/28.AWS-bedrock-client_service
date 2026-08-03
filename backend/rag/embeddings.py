"""
Función de embeddings explícita, usando un modelo open source de
Hugging Face vía la librería sentence-transformers.
"""

from chromadb.utils import embedding_functions

# Modelo open source de Hugging Face (~80MB, corre en CPU sin problema).
# Genera vectores de 384 dimensiones. Es liviano y rápido — ideal para
# desarrollo y para la mayoría de casos de RAG que no requieren
# precisión de nivel investigación.
#
# Para ver otras opciones: https://huggingface.co/models?library=sentence-transformers
# Nombre del modelo de Embedings de Huggieng Face
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function():
    """
    Devuelve una "embedding function" compatible con ChromaDB.
    La primera vez que se llama, descarga el modelo de Hugging Face
    (~80MB) y lo cachea localmente en ~/.cache/huggingface — las
    siguientes ejecuciones son instantáneas, sin volver a descargar.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)