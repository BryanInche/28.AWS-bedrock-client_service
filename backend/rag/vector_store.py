"""
Wrapper sobre ChromaDB. Aísla toda la lógica de la base de datos
vectorial en un solo lugar, para que el resto del backend (el futuro
agente, el MCP server) nunca tenga que saber cómo funciona Chroma por
dentro — solo llama a add_documents() y search().
"""

import chromadb

from embeddings import get_embedding_function

# Carpeta donde Chroma guarda la base de datos EN DISCO.
CHROMA_PATH = "./chroma_data"

# Nombre de la colección. Una "colección" en Chroma es como una tabla
# en una base de datos relacional: un espacio con nombre donde viven
# documentos relacionados.
COLLECTION_NAME = "base_de_conocimiento"


class VectorStore:
    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=path)

        # Ahora pasamos explícitamente nuestra función de embeddings de
        # Hugging Face, en vez de dejar que Chroma use la suya por
        # defecto de forma implícita.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
        )

    def add_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """
        Agrega documentos a la base vectorial.

        texts:     el contenido real (ej. un párrafo, o el texto OCR de una imagen)
        metadatas: info adicional por documento (ej. {"source": "manual.pdf", "page": 3})
        ids:       identificador único por documento (tú lo defines, ej. "doc_1")
        """
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 3) -> dict:
        """
        Busca los documentos más relevantes para una consulta, por
        similitud semántica (no por coincidencia exacta de palabras).
        """
        return self.collection.query(query_texts=[query], n_results=n_results)

    def count(self) -> int:
        """Cuántos documentos hay guardados actualmente. Útil para debug."""
        return self.collection.count()