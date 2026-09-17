"""
Sube el adaptador LoRA ya entrenado (fine_tuning/lora_bryan) a tu
cuenta de Hugging Face Hub, para que el backend en producción pueda
descargarlo sin depender de la carpeta local ni de Docker.

Requisito: tener HF_TOKEN en tu .env, con permiso "Write".
"""

import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

# Lee tu archivo .env local y carga HF_TOKEN a las variables de
# entorno
load_dotenv()

# Carpeta LOCAL donde guardaste el resultado del entrenamiento
CARPETA_ADAPTADOR = "lora_bryan"

# Nombre del repositorio que se creará/usará en Hugging Face, con el
# formato obligatorio "usuario/nombre-del-repo".
NOMBRE_REPO = "Bryaninchecondor/lora-bryan-atencion-cliente-es"


def subir_adaptador():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Falta HF_TOKEN en tu .env")

    # Verifica que la carpeta del adaptador realmente exista
    if not os.path.isdir(CARPETA_ADAPTADOR):
        raise ValueError(f"No se encontró la carpeta: {CARPETA_ADAPTADOR}")

    # HfApi es el "cliente" que habla con los servidores de Hugging
    # Face — todas las operaciones (crear repo, subir archivos) pasan
    # por este objeto, autenticado con el token.
    api = HfApi(token=token)

    print(f"Creando o confirmando el repositorio: {NOMBRE_REPO}...")

    # Crea el repositorio remoto en tu cuenta de Hugging Face.
    api.create_repo(repo_id=NOMBRE_REPO, repo_type="model", exist_ok=True, private=True)

    print(f"Subiendo archivos de {CARPETA_ADAPTADOR}/ a {NOMBRE_REPO}...")

    # Sube TODOS los archivos de la carpeta local al repo remoto, tal
    # cual están 
    api.upload_folder(
        folder_path=CARPETA_ADAPTADOR,
        repo_id=NOMBRE_REPO,
        repo_type="model",
    )

    print(f"\n Listo. Tu adaptador ahora vive en: https://huggingface.co/{NOMBRE_REPO}")
    print(f"   Para usarlo, configura en tu .env del backend:")
    print(f"   LORA_PATH={NOMBRE_REPO}")


if __name__ == "__main__":
    subir_adaptador()