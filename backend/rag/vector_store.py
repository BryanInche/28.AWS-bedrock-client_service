"""
Wrapper sobre ChromaDB. Aísla toda la lógica de la base de datos
vectorial en un solo lugar, para que el resto del backend (el futuro
agente, el MCP server) nunca tenga que saber cómo funciona Chroma por
dentro — solo llama a add_documents() y search().
"""

import chromadb

from embeddings import get_embedding_function

# Carpeta donde Chroma guarda la base de datos EN DISCO.
CHROMA_PATH = "./chroma_data_bryan"

# Nombre de la colección. Una "colección" en Chroma es como una tabla
# en una base de datos relacional: un espacio con nombre donde viven
# documentos relacionados.
COLLECTION_NAME = "base_de_conocimiento_vector"


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
        Agrega documentos NUEVOS. Si algún "id" ya existe en la
        colección, Chroma Db lanza un error — úsalo solo cuando se este
        seguro de que los ids son nuevos.
        """
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def upsert_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """
        "Upsert" = update + insert. Si el "id" ya existe, ACTUALIZA ese
        documento (sobrescribe texto/metadata/vector). Si no existe, lo
        crea. Esto usar para ingesta de archivos
        reales: si vuelves a correr el mismo PDF, actualiza en vez de
        duplicar, siempre que los ids sean determinísticos (ver
        ingesta.py: se generan a partir de la ruta + página + posición
        del chunk, no al azar).
        """
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 3) -> dict:
        """
        Busca los documentos más relevantes para una consulta, por
        similitud semántica (no por coincidencia exacta de palabras).
        """
        return self.collection.query(query_texts=[query], n_results=n_results)

    def count(self) -> int:
        """Cuántos documentos hay guardados actualmente. Útil para debug."""
        return self.collection.count()

    def delete_by_source(self, source: str) -> None:
        """
        Elimina todos los chunks que vinieron de un archivo específico.
        Útil para borras/reemplazas un PDF y quieres limpiar sus
        fragmentos viejos de la base antes de re-ingerirlo.
        """
        self.collection.delete(where={"source": source})