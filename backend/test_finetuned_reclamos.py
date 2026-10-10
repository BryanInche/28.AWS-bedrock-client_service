"""
Prueba del FinetunedProvider con varios reclamos de ejemplo
"""

import os

from dotenv import load_dotenv

load_dotenv()

from llm_proveedors.finetuned_llm import FinetunedProvider

# Reclamos NUEVOS, con redacción distinta a los del dataset original,
# para confirmar generalización, no memorización.
RECLAMOS_DE_PRUEBA = [
    "Compré un producto hace 3 días y llegó dañado, ¿qué hago?",
    "Necesito cancelar mi pedido número 4521, ya no lo quiero",
    "El repartidor nunca llegó y ya pasaron 2 semanas",
    "Me cobraron dos veces por la misma compra",
]

if __name__ == "__main__":
    print("Cargando el modelo fine-tuneado (puede tardar unos segundos)...\n")
    provider = FinetunedProvider()

    for i, reclamo in enumerate(RECLAMOS_DE_PRUEBA, start=1):
        print(f"--- Reclamo {i} ---")
        print(f"Usuario: {reclamo}")
        respuesta = provider.call_text(reclamo)
        print(f"Agente:  {respuesta}\n")