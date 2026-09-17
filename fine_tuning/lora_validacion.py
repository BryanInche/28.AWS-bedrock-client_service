import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # descarga el modelo GRANDE, del repo de DeepSeek
#LORA_PATH = "./adapters/lora_bryan" # O "tu_usuario_hf/lora_bryan"
LORA_PATH = "./lora_bryan" # carga SOLO adaptador CHICO, local

print("Cargando modelo base y tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Aplicando adaptador LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_PATH) # ← los COMBINA en memoria, ModeloBase + Adaptador

# Realizar una inferencia de prueba
prompt = "Tengo un reclamo sobre un producto que me enviaron ayer, me pueden ayudar?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))