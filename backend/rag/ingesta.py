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
from pathlib import Path

from pypdf import PdfReader

from vector_store import VectorStore

CHUNK_SIZE = 800  # Numero de chunks rescatados para particion
CHUNK_OVERLAP = 150 # Numero de chunks que se traen del anterior bloque

# Carpeta donde viven todos los PDFs a ingerir, por convención.
CARPETA_DOCUMENTOS = Path(r"C:\Freelance\AWS_Bedrock\backend\rag\documentos_fuentes")

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
        # Se agregan los documentos al Vector Store
        store.upsert_documents(texts=textos, metadatas=metadatas, ids=ids)

    return len(textos)


def ingesta_carpeta_completa(carpeta: Path, store: VectorStore) -> None:
    """
    Recorre TODOS los archivos .pdf dentro de "carpeta" y los ingiere
    uno por uno. glob("*.pdf") devuelve la lista de archivos que
    terminan en .pdf dentro de esa carpeta (no busca en subcarpetas).
    """
    pdfs_encontrados = list(carpeta.glob("*.pdf"))
 
    if not pdfs_encontrados:
        print(f"No se encontraron PDFs en: {carpeta}")
        return
 
    print(f"Se encontraron {len(pdfs_encontrados)} PDF(s) en {carpeta}:\n")
 
    for ruta_pdf in pdfs_encontrados:
        cantidad = ingesta_pdf(ruta_pdf, store)
        print(f" {ruta_pdf.name}: {cantidad} fragmentos procesados")
 


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # 1. Se ejecuta __init__() de la clase VectorStore (en vector_store.py).
    # 2. Chroma revisa la carpeta "./chroma_data_bryan":
    #       - ¿Existe ya, de una ejecución anterior tuya? -> la ABRE
    #         con todo lo que ya tenía guardado dentro.
    #       - ¿Es la primera vez que corres cualquier script de RAG? ->
    #         Chroma CREA esa carpeta desde cero, vacía.
    # 3. En ninguno de los dos casos se borra nada: "store" es solo
    #    tu "conexión" para leer/escribir en esa base de datos, igual
    #    que abrir una conexión a MySQL/PostgreSQL.
    # ---------------------------------------------------------------
    # A partir de esta línea, "store" ya tiene disponibles los métodos
    # upsert_documents(), search() y count() para usar abajo.
    # ---------------------------------------------------------------

    store = VectorStore()
 
    if len(sys.argv) >= 2:
        # Se pasó una ruta específica como argumento: solo ese archivo.
        ruta = Path(sys.argv[1])
        print(f"--- Ingiriendo un solo archivo: {ruta} ---")
        cantidad = ingesta_pdf(ruta, store)
        print(f"Se procesaron {cantidad} fragmentos.")
    else:
        # Sin argumentos: procesa TODOS los PDFs de documentos_fuentes/
        print(f"--- Ingiriendo todos los PDFs de: {CARPETA_DOCUMENTOS} ---\n")
        ingesta_carpeta_completa(CARPETA_DOCUMENTOS, store)