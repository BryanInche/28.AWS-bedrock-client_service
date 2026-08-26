"""
Convierte data/dataset_es.jsonl al
formato final que espera el entrenamiento: UN SOLO campo "text" por
ejemplo, combinando instrucción + respuesta con una plantilla fija.
También separa los datos en train (para entrenar) y test (para medir
después si el modelo realmente aprendió, con datos que nunca vio).

Requisito previo: haber corrido translate_dataset.py antes.

Uso:
    python build_training_file.py
"""

import json
import random

ARCHIVO_ENTRADA = "data/dataset_es.jsonl"
ARCHIVO_TRAIN = "data/train.jsonl"
ARCHIVO_TEST = "data/test.jsonl"

# Qué porcentaje se reserva para evaluación
PORCENTAJE_TEST = 0.2  # 20% para test, 80% para entrenar

SEMILLA_ALEATORIA = 42  # fija, para que el split train/test sea siempre el mismo si vuelves a ejecutar


def construir_texto_entrenamiento(prompt: str, completion: str) -> str:
    """
    Combina la instrucción y la respuesta en UN SOLO texto, usando una
    plantilla fija. El modelo aprende a "completar" este patrón: dado
    todo lo que está antes de "### Respuesta:", debe generar lo que
    sigue.

    Esta plantilla (estilo "Alpaca") es una convención muy usada en
    fine-tuning de modelos de instrucciones no es obligatoria, pero
    sí debe ser SIEMPRE la misma en entrenamiento e inferencia después.
    """
    return f"""### Instrucción:
{prompt}

### Respuesta:
{completion}"""


def cargar_dataset_traducido() -> list[dict]:
    filas = []
    with open(ARCHIVO_ENTRADA, encoding="utf-8") as f:
        for linea in f:
            filas.append(json.loads(linea))
    return filas


def construir_archivo_entrenamiento_fine_tuning():
    print(f"Leyendo: {ARCHIVO_ENTRADA}")
    filas = cargar_dataset_traducido()
    print(f"Se cargaron {len(filas)} filas.\n")

    # Construimos el campo "text" para cada fila
    filas_procesadas = []
    for row in filas:
        texto_final = construir_texto_entrenamiento(row["prompt"], row["completion"])
        filas_procesadas.append({
            "text": texto_final,
            "category": row["category"],  # se conserva por si la necesitamos después
            "intent": row["intent"],
        })

    # Mezclamos de forma reproducible, y separamos train/test
    random.seed(SEMILLA_ALEATORIA)
    random.shuffle(filas_procesadas)  # Para unir todo

    punto_de_corte = int(len(filas_procesadas) * (1 - PORCENTAJE_TEST))
    filas_train = filas_procesadas[:punto_de_corte]
    filas_test = filas_procesadas[punto_de_corte:]

    with open(ARCHIVO_TRAIN, "w", encoding="utf-8") as f:
        for fila in filas_train:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    with open(ARCHIVO_TEST, "w", encoding="utf-8") as f:
        for fila in filas_test:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    print(f"Train: {len(filas_train)} ejemplos -> {ARCHIVO_TRAIN}")
    print(f"Test:  {len(filas_test)} ejemplos -> {ARCHIVO_TEST}\n")

    print("--- EJEMPLO de cómo queda un registro de entrenamiento ---")
    print(filas_train[0]["text"])


if __name__ == "__main__":
    construir_archivo_entrenamiento_fine_tuning()