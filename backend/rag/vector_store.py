"""
Esto conecta con la base de datos vectorial (Chroma) y expone solo
dos metodos: guardar texto (upsert_documents) y buscar (search).
"""
from pathlib import Path

# Libreria para manejar Base de Datos Vectorial
import chromadb

# Embeding de Huggieng Face
from embeddings import get_embedding_function

# Carpeta donde Chroma guarda la base de datos EN DISCO (persistente).
# Si esta carpeta no existe, Chroma la crea sola la primera vez.
#CHROMA_PATH = "./chroma_data_bryan"

# AHORA (absoluto, anclado al archivo, funciona desde CUALQUIER carpeta):
CHROMA_PATH = str(Path(__file__).parent / "chroma_data_bryan")

# Nombre de la "tabla" (colección) donde viven todos tus documentos.
COLLECTION_NAME = "base_de_conocimiento_vector"


class VectorStore:
    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        # PersistentClient: abre (o crea, si es la primera vez) la
        # base de datos guardada en la carpeta CHROMA_PATH.
        self.client = chromadb.PersistentClient(path=path)

        # get_or_create_collection:
        #   - Si la colección "base_de_conocimiento_vector" YA EXISTE
        #     en disco (de una ejecución anterior) -> la abre tal cual
        #     está, con todo lo que ya tenía guardado.
        #   - Si NO EXISTE todavía (primera vez que corres esto) -> la
        #     crea vacía.
        # embedding_function: el modelo (definido en embeddings.py)
        # que convierte cada texto en un vector antes de guardarlo.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
        )

    def upsert_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """
        Guarda texto en la base. Si el "id" ya existía, lo actualiza
        (sobrescribe); si no existía, lo crea. Por eso es seguro volver
        a correr la ingesta del mismo PDF sin generar duplicados.
        """
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 3) -> dict:
        """Busca los "n_results" documentos más parecidos a "query" (por significado, no por palabras exactas)."""
        return self.collection.query(query_texts=[query], n_results=n_results)

    def count(self) -> int:
        """Cuántos documentos hay guardados actualmente."""
        return self.collection.count()