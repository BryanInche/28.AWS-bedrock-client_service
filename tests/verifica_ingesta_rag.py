"""
Verifica que los PDFs realmente se guardaron en la base vectorial.

Uso:
    python tests/verificar_rag.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "backend" / "rag"))

from vector_store import VectorStore

if __name__ == "__main__":
    store = VectorStore()

    # 1. Verificar que documentos hay guardados
    total = store.count()
    print(f"Documentos guardados en la base vectorial: {total}")

    if total == 0:
        print("No hay nada guardado. Corre 'python rag/ingesta.py' primero.")
        sys.exit(1)

    # 2. Una búsqueda simple, para confirmar que el contenido es real
    #    y que la búsqueda semántica encuentra algo coherente.
    #resultado = store.search("¿de qué tratan estos documentos?", n_results=1)
    resultado = store.search("En una regex podemos usar clases de expresiones regulares", n_results=1)

    texto = resultado["documents"][0][0]
    archivo = resultado["metadatas"][0][0]["source"]

    print(f"\nEjemplo de contenido encontrado:")
    print(f"Archivo: {archivo}")
    print(f"Texto: {texto[:150]}...")
    print("\n La ingesta funcionó correctamente.")