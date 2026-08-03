"""
Ingesta de documentos PDF a la base de datos vectorial (Chroma Vector Store).

Uso:
    python rag/ingestion.py ruta/al/documento.pdf
"""

import sys
import uuid

from pypdf import PdfReader

from vector_store import VectorStore

# Tamaño máximo de cada "chunk" (fragmento) de texto, en caracteres.
# ¿Por qué tokenizar el PDF en vez de guardarlo entero?
# 1. Los modelos de embeddings tienen un límite de tokens de entrada.
# 2. Fragmentos más pequeños dan búsquedas más precisas: si guardas un
#    PDF de 50 páginas como UN solo documento, buscar "política de
#    reembolsos" te devolvería el PDF entero, no el párrafo relevante.
CHUNK_SIZE = 800    # Numero de chunks rescatados
CHUNK_OVERLAP = 100  # caracteres compartidos entre chunks consecutivos,
                      # para no cortar una idea justo a la mitad


def extraer_texto_pdf(ruta_pdf: str) -> list[tuple[str, int]]:
    """
    Extrae el texto de cada página de un PDF.
    Devuelve una lista de tuplas (texto_de_la_pagina, numero_de_pagina).
    """
    reader = PdfReader(ruta_pdf)
    paginas = []
    for i, page in enumerate(reader.pages):
        texto = page.extract_text()
        if texto and texto.strip():  # ignora páginas vacías (ej. solo imágenes)
            paginas.append((texto, i + 1))
    return paginas


def particionar_texto(texto: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide un texto largo en fragmentos más pequeños, con solapamiento
    entre ellos para no perder contexto en los bordes.
    """
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        chunks.append(texto[inicio:fin])
        inicio += chunk_size - overlap
    return chunks


def ingesta_pdf(ruta_pdf: str, store: VectorStore) -> int:
    """
    Extrae, particiona, y guarda un PDF completo en la base vectorial Qroma.
    Devuelve la cantidad de chunks agregados.
    """
    paginas = extraer_texto_pdf(ruta_pdf)

    textos, metadatas, ids = [], [], []

    for texto_pagina, num_pagina in paginas:
        for chunk in particionar_texto(texto_pagina):
            if not chunk.strip():
                continue
            textos.append(chunk)
            metadatas.append({
                "source": ruta_pdf,
                "page": num_pagina,
                "type": "pdf",
            })
            # ID único por chunk, para evitar colisiones si ingieres
            # el mismo PDF más de una vez con distinto contenido.
            ids.append(str(uuid.uuid4()))

    if textos:
        store.add_documents(texts=textos, metadatas=metadatas, ids=ids)

    return len(textos)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ingestion.py ruta/al/documento.pdf")
        sys.exit(1)

    ruta = sys.argv[1]
    store = VectorStore()

    print(f"--- Ingiriendo: {ruta} ---")
    cantidad = ingesta_pdf(ruta, store)
    print(f"Se agregaron {cantidad} fragmentos a la base vectorial.")
    print(f"Total de documentos en la colección: {store.count()}")