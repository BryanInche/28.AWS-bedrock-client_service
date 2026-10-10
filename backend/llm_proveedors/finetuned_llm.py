"""
Proveedor que usa el modelo fine-tuneado (DeepSeek + adaptador LoRA).
Mismo contrato LLMProvider que bedrock_llm.py y groq_open_llm.py.

LORA_PATH acepta 2 formas (PeftModel las maneja igual):
  - Carpeta local:     ../fine_tuning/lora_bryan
  - Repo del Hub:      Bryaninchecondor/lora-bryan-atencion-cliente-es
"""

import os
from .base import LLMProvider


# Define la ruta por defecto donde está guardado el adaptador LoRA local
BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_LORA_PATH = os.path.join(
    #"..": sube a backend. "..": sube a la raíz del proyecto
    os.path.dirname(__file__), "..", "..", "fine_tuning", "lora_bryan"
)


class FinetunedProvider(LLMProvider):
    def __init__(self, lora_path: str | None = None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        lora_path = lora_path or os.getenv("LORA_PATH", DEFAULT_LORA_PATH)
        hf_token = os.getenv("HF_TOKEN")  # necesario: tu repo del Hub es privado

        print(f"[FinetunedProvider] Cargando modelo base: {BASE_MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

        #base_model = AutoModelForCausalLM.from_pretrained(
        #    BASE_MODEL_ID,
        #    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        #    device_map="auto",
        #)

        hay_gpu = torch.cuda.is_available()
 
        if hay_gpu:
            # Con GPU real (ej. Kaggle): device_map="auto" reparte el
            # modelo entre GPU/CPU automáticamente — tiene sentido aquí.
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.float16,
                device_map="auto",
            )
        else:
            # SIN GPU (tu laptop): cargamos directo a CPU, sin
            # device_map. Esto evita que "accelerate" intente repartir
            # el modelo entre dispositivos 
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.float32,
            )
            base_model = base_model.to("cpu")

        print(f"[FinetunedProvider] Aplicando adaptador LoRA: {lora_path}...")
        self.model = PeftModel.from_pretrained(base_model, lora_path, token=hf_token)
        self.model.eval()

    def call_text(self, prompt: str, model: str | None = None) -> str:
        # Misma plantilla usada al entrenar (3data_preparation.py).
        texto_entrada = f"### Instrucción:\n{prompt}\n\n### Respuesta:\n"
        inputs = self.tokenizer(texto_entrada, return_tensors="pt").to(self.model.device)

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        texto = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # El modelo devuelve prompt + respuesta juntos; nos quedamos
        # solo con lo que viene después del marcador.
        if "### Respuesta:" in texto:
            return texto.split("### Respuesta:")[-1].strip()
        return texto.strip()

    def call_image(self, image_bytes: bytes, filename: str, caption: str, model: str | None = None) -> str:
        raise NotImplementedError(
            "FinetunedProvider no soporta imágenes (solo se entrenó con texto). "
            "Usa GroqProvider o BedrockProvider."
        )