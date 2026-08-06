"""
Ingesta de documentos PDF a la base de datos vectorial (Chroma).

Uso:
    python ingesta.py ruta/al/documento.pdf

Seguro de correr varias veces con el mismo archivo: los IDs de cada
chunk se calculan a partir de (archivo + página + posición), así que
re-ingerir el mismo PDF actualiza sus chunks en vez de duplicarlos.
"""

import hashlib
import sys

from pypdf import PdfReader

from vector_store import VectorStore

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def extraer_texto_pdf(ruta_pdf: str) -> list[tuple[str, int]]:
    """Lee el PDF y devuelve el texto de cada página junto con su número de página."""
    reader = PdfReader(ruta_pdf)
    paginas = []
    for i, page in enumerate(reader.pages):
        texto = page.extract_text()
        if texto and texto.strip():
            paginas.append((texto, i + 1))
    return paginas


def particionar_texto(texto: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide un texto largo en fragmentos más chicos (chunks), para que la búsqueda sea más precisa."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        chunks.append(texto[inicio:fin])
        inicio += chunk_size - overlap
    return chunks


def generar_id_deterministico(source: str, pagina: int, posicion_chunk: int) -> str:
    """Genera siempre el MISMO id para la misma combinación (archivo, página, posición)."""
    contenido = f"{source}|page={pagina}|chunk={posicion_chunk}"
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


def ingesta_pdf(ruta_pdf: str, store: VectorStore) -> int:
    """Extrae, particiona y guarda un PDF completo en la base vectorial. Devuelve cuántos chunks se guardaron."""
    paginas = extraer_texto_pdf(ruta_pdf)

    textos, metadatas, ids = [], [], []

    for texto_pagina, num_pagina in paginas:
        chunks = particionar_texto(texto_pagina)
        for posicion, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            textos.append(chunk)
            metadatas.append({"source": ruta_pdf, "page": num_pagina, "type": "pdf"})
            ids.append(generar_id_deterministico(ruta_pdf, num_pagina, posicion))

    if textos:
        store.upsert_documents(texts=textos, metadatas=metadatas, ids=ids)

    return len(textos)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ingesta.py ruta/al/documento.pdf")
        sys.exit(1)

    ruta = sys.argv[1]

    # ---------------------------------------------------------------
    # QUÉ PASA EXACTAMENTE AQUÍ, EN "store = VectorStore()":
    #
    # 1. Se ejecuta __init__() de la clase VectorStore (en vector_store.py).
    # 2. Chroma revisa la carpeta "./chroma_data_bryan":
    #       - ¿Existe ya, de una ejecución anterior tuya? -> la ABRE
    #         con todo lo que ya tenía guardado dentro.
    #       - ¿Es la primera vez que corres cualquier script de RAG? ->
    #         Chroma CREA esa carpeta desde cero, vacía.
    # 3. En ninguno de los dos casos se borra nada: "store" es solo
    #    tu "conexión" para leer/escribir en esa base de datos, igual
    #    que abrir una conexión a MySQL/PostgreSQL — no crea una base
    #    nueva cada vez que la abres, solo te conectas a la que existe
    #    (o la crea si es la primera vez).
    #
    # A partir de esta línea, "store" ya tiene disponibles los métodos
    # upsert_documents(), search() y count() para usar abajo.
    # ---------------------------------------------------------------
    store = VectorStore()

    print(f"--- Ingiriendo: {ruta} ---")
    cantidad = ingesta_pdf(ruta, store)
    print(f"Se procesaron {cantidad} fragmentos (creados o actualizados).")
    print(f"Total de documentos en la colección: {store.count()}")