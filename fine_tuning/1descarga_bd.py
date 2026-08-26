"""
Descarga el dataset original (en inglés) y lo guarda en disco, SIN
traducir nada todavía. El objetivo es que puedas inspeccionar la
estructura real (columnas, tipos de dato, ejemplos) antes de decidir
qué hacer con cada campo.

Uso:
    python download_dataset.py
"""

import json

from datasets import load_dataset

CANTIDAD_EJEMPLOS = 50  # mismo tamaño de prueba de antes
ARCHIVO_SALIDA = "data/raw_dataset_en.jsonl"


def descargar_dataset():
    print("Descargando dataset desde Hugging Face...\n")

    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split=f"train[:{CANTIDAD_EJEMPLOS}]",
    )

    # =====================================================================
    # 1. ESTRUCTURA DEL DATASET: qué columnas tiene, y de qué tipo es cada una
    # =====================================================================
    print("=" * 70)
    print("ESTRUCTURA DEL DATASET (columnas y tipos)")
    print("=" * 70)
    print(dataset.features)
    print()

    # =====================================================================
    # 2. CUÁNTOS EJEMPLOS DESCARGAMOS
    # =====================================================================
    print(f"Cantidad de filas descargadas: {len(dataset)}\n")

    # =====================================================================
    # 3. UN EJEMPLO COMPLETO, CRUDO, TAL CUAL VIENE (fila 0)
    # =====================================================================
    print("=" * 70)
    print("EJEMPLO COMPLETO (fila 0, sin modificar)")
    print("=" * 70)
    for columna, valor in dataset[0].items():
        print(f"{columna}: {valor}")
    print()

    # =====================================================================
    # 4. VALORES ÚNICOS de "category" e "intent", para entender cuántas
    #    clases distintas existen (importante para la tarea de clasificación)
    # =====================================================================
    categorias_unicas = sorted(set(dataset["category"]))
    intents_unicos = sorted(set(dataset["intent"]))

    print("=" * 70)
    print(f"CATEGORÍAS ÚNICAS encontradas en estos {CANTIDAD_EJEMPLOS} ejemplos ({len(categorias_unicas)}):")
    print("=" * 70)
    print(categorias_unicas)
    print()

    print("=" * 70)
    print(f"INTENTS ÚNICOS encontrados en estos {CANTIDAD_EJEMPLOS} ejemplos ({len(intents_unicos)}):")
    print("=" * 70)
    print(intents_unicos)
    print()

    # =====================================================================
    # 5. GUARDAR TAL CUAL EN DISCO (sin traducir), para inspeccionarlo
    #    con calma, o para que translate_dataset.py lo lea después SIN
    #    tener que volver a descargar de Hugging Face cada vez.
    # =====================================================================
    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    print(f"Dataset original guardado en: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    descargar_dataset()