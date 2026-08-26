"""
Traduce el dataset local (ya descargado por download_dataset.py) de
inglés a español, usando un modelo de traducción LOCAL.

Requisito previo: haber corrido "python download_dataset.py" antes,
para que exista data/raw_dataset_en.jsonl

Uso:
    python translate_dataset.py
"""

import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELO_TRADUCCION = "Helsinki-NLP/opus-mt-en-es"

ARCHIVO_ENTRADA = "data/raw_dataset_en.jsonl"
ARCHIVO_SALIDA = "data/dataset_es.jsonl"


def limpiar_placeholders(texto: str) -> str:
    """Convierte {{Variable}} a [Variable] antes de traducir, para no confundir al modelo."""
    return texto.replace("{{", "[").replace("}}", "]")


def cargar_dataset_local() -> list[dict]:
    """Lee el archivo JSONL que guardó download_dataset.py, línea por línea."""
    filas = []
    with open(ARCHIVO_ENTRADA, encoding="utf-8") as f:
        for linea in f:
            filas.append(json.loads(linea))
    return filas


def traducir_texto(texto: str, tokenizer, model) -> str:
    """
    Traduce un texto usando el modelo cargado directamente, SIN pasar
    por pipeline(). Esto evita depender de que "translation" esté
    registrado como tarea de alto nivel en la versión de transformers
    instalada (eso fue justo lo que falló antes).

    El proceso manual son 3 pasos, siempre los mismos para cualquier
    modelo tipo "seq2seq" (traducción, resumen, etc.):
      1. tokenizer(...)   -> convierte texto a números que el modelo entiende
      2. model.generate() -> el modelo genera la traducción (en números)
      3. tokenizer.decode() -> convierte esos números de vuelta a texto
    """
    entrada = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
    salida = model.generate(**entrada, max_length=512)
    return tokenizer.decode(salida[0], skip_special_tokens=True)


def traducir_dataset():
    print(f"Leyendo dataset local desde: {ARCHIVO_ENTRADA}")
    filas = cargar_dataset_local()
    print(f"Se cargaron {len(filas)} filas (sin volver a descargar de Hugging Face).\n")

    print("Cargando modelo de traducción (la primera vez descarga ~300MB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_TRADUCCION)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODELO_TRADUCCION)

    filas_traducidas = []

    for i, row in enumerate(filas):
        instruction_en = limpiar_placeholders(row["instruction"])
        response_en = limpiar_placeholders(row["response"])

        prompt_es = traducir_texto(instruction_en, tokenizer, model)
        completion_es = traducir_texto(response_en, tokenizer, model)

        if i == 0:
            print("\n--- EJEMPLO (fila 0) ---")
            print(f"INSTRUCTION (en): {instruction_en}")
            print(f"PROMPT (es):      {prompt_es}\n")
            print(f"RESPONSE (en, primeros 200 caracteres): {response_en[:200]}...")
            print(f"COMPLETION (es, primeros 200 caracteres): {completion_es[:200]}...\n")

        filas_traducidas.append({
            "prompt": prompt_es,
            "completion": completion_es,
            "category": row["category"],
            "intent": row["intent"],
        })

        if (i + 1) % 10 == 0:
            print(f"Traducidas {i + 1}/{len(filas)}...")

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        for fila in filas_traducidas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    print(f"\nDataset traducido guardado en: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    traducir_dataset()