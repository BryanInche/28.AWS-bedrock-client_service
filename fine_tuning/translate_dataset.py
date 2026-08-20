import json
from datasets import load_dataset
from deep_translator import GoogleTranslator

def inspect_and_translate():
    print("Descargando dataset desde Hugging Face...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train[:50]")
    
    # -------------------------------------------------------------
    # PASO 1: INSPECCIONAR DATA CRUDA
    # -------------------------------------------------------------
    raw_sample = dataset[0]
    print("\n" + "="*70)
    print("1. DATA CRUDA (Como viene de Hugging Face en formato Diccionario/Tabla):")
    print("="*70)
    print(json.dumps(raw_sample, indent=2, ensure_ascii=False))
    
    translator = GoogleTranslator(source='en', target='es')
    output_file = "reclamos_es_sample.jsonl"
    
    print("\nTraduciendo y transformando...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            inst_en = row['instruction']
            resp_en = row['response']
            
            # Traducción
            inst_es = translator.translate(inst_en)
            resp_es = translator.translate(resp_en)
            
            # Estructuración para Fine-Tuning
            prompt_estructurado = f"Eres un agente de servicio al cliente experto y empático. Resuelve el siguiente caso:\n\nCliente: {inst_es}"
            
            json_line = {
                "prompt": prompt_estructurado,
                "completion": resp_es
            }
            
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            
            # -------------------------------------------------------------
            # PASO 2 Y 3: INSPECCIONAR TRANSFORMACIÓN
            # -------------------------------------------------------------
            if i == 0:
                print("\n" + "="*70)
                print("2. DATA ESTRUCTURADA EN JSONL (Formato Intermedio):")
                print("="*70)
                print(json.dumps(json_line, indent=2, ensure_ascii=False))
                
                print("\n" + "="*70)
                print("3. DATA FINAL EN FORMATO CHATML (Como la lee DeepSeek internamente):")
                print("="*70)
                chatml_preview = f"<|im_start|>user\n{prompt_estructurado}<|im_end|>\n<|im_start|>assistant\n{resp_es}<|im_end|>"
                print(chatml_preview)
                print("="*70 + "\n")

    print(f"¡Proceso completado! Revisa el archivo '{output_file}'.")

if __name__ == "__main__":
    inspect_and_translate()