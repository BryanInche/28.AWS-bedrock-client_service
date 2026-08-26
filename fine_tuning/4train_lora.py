"""
Entrena un adaptador LoRA sobre DeepSeek-R1-Distill-Qwen-1.5B, usando
el dataset de atención al cliente en español (train.jsonl).

IMPORTANTE: este script necesita GPU real para completar el
entrenamiento — está pensado para correr en Kaggle, NO en tu laptop.
En tu laptop, como mucho, sirve para detectar errores de sintaxis
antes de subirlo (no vas a poder terminar un entrenamiento real sin
GPU).

Cómo correrlo en Kaggle:
    1. Crea un notebook nuevo, activa GPU (Settings -> Accelerator -> GPU T4 x2)
    2. Agrega tu HF_TOKEN en Add-ons -> Secrets
    3. En la primera celda:
        !git clone https://github.com/TU_USUARIO/TU_REPO.git
        %cd TU_REPO/fine_tuning
        !pip install -r requirements.txt
    4. En la siguiente celda:
        !python train_lora.py
"""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

ARCHIVO_TRAIN = "data/train.jsonl"
CARPETA_SALIDA = "adapters/lora_bryan"


def obtener_token_hf() -> str:
    """
    Busca el token de Hugging Face en 2 lugares posibles, en este orden:
    1. Kaggle Secrets (cuando corre en Kaggle)
    2. Variable de entorno HF_TOKEN (cuando corre en tu local, vía .env)
    """
    try:
    # Opcion 1: busqueda de clave en servidor kagle
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    # Opcion 2: busqueda en local
    except ImportError:
        # No estamos en Kaggle (kaggle_secrets no existe fuera de ahí)
        from dotenv import load_dotenv

        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("No se encontró HF_TOKEN ni en Kaggle Secrets ni en .env local.")
        return token


def cargar_modelo_y_tokenizer(token: str):
    """
    Carga el modelo SIN cuantización (precisión bf16 completa) — esto
    es lo que distingue a LoRA de QLoRA. Requiere más memoria GPU que
    la versión QLoRA, pero es más simple y típicamente da resultados
    ligeramente mejores.
    """

    # El tokenizador convierte texto en números (IDs) que el modelo 
    # entiende, y viceversa. Cada modelo tiene su propio vocabulario — 
    # por eso se descarga junto con el modelo, no es genérico.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # descarga los ~1.5 mil millones de parámetros ya entrenados de DeepSeek, en formato bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def construir_lora_config() -> LoraConfig:
    """
    Configuración del adaptador LoRA. "r" (rank) controla cuántos
    parámetros nuevos se entrenan — más alto = más capacidad de
    aprendizaje, pero más memoria y riesgo de sobreajuste.
    """
    return LoraConfig(
    r=16,                # el "rank" de las matrices A y B — más alto = más capacidad, más memoria
    lora_alpha=16,        # factor de escala del ajuste LoRA (alpha/r en la fórmula de arriba)
    lora_dropout=0,        # sin dropout (regularización) en las capas LoRA
    bias="none",             # no se entrenan los términos de sesgo, solo A y B
    task_type="CAUSAL_LM",    # le dice a la librería que es un modelo de lenguaje autoregresivo (predice el siguiente token)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def entrenar():
    token = obtener_token_hf()

    print(f"Cargando modelo: {MODEL_ID} (LoRA, sin cuantización)...")
    model, tokenizer = cargar_modelo_y_tokenizer(token)

    print(f"Cargando dataset de entrenamiento: {ARCHIVO_TRAIN}...")
    train_dataset = load_dataset("json", data_files=ARCHIVO_TRAIN, split="train")
    print(f"Ejemplos de entrenamiento: {len(train_dataset)}")

    lora_config = construir_lora_config()

    training_args = SFTConfig(
        output_dir=CARPETA_SALIDA,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        bf16=True,
        dataset_text_field="text",  # el dataset ya trae el campo "text" listo, de build_training_file.py
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
        processing_class=tokenizer,
    )

    print("\nIniciando entrenamiento LoRA...\n")
    resultado = trainer.train()
    print("\nEntrenamiento terminado:", resultado)

    # guarda solo las matrices LoRA (unos pocos MB), no el modelo completo
    print(f"\nGuardando adaptador en: {CARPETA_SALIDA}")
    trainer.model.save_pretrained(CARPETA_SALIDA)
    tokenizer.save_pretrained(CARPETA_SALIDA)

    print("Listo. El adaptador LoRA está guardado y listo para evaluar.")


if __name__ == "__main__":
    entrenar()