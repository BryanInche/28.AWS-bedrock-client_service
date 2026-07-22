# =====================================================================
# Dockerfile - Backend FastAPI (AWS Bedrock)
# =====================================================================
# Usamos una imagen "slim" en vez de la imagen completa de Python:
# pesa mucho menos y reduce la superficie de ataque en producción.
FROM python:3.12-slim

# Evita que Python genere archivos .pyc y fuerza logs sin buffer
# (para que aparezcan en tiempo real en "docker logs").
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos SOLO requirements.txt primero. Esto es una técnica de cache
# de capas de Docker: si el código cambia pero las dependencias no,
# Docker reutiliza esta capa y el build es mucho más rápido.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Ahora sí copiamos el resto del código del backend.
COPY . .

# Buenas prácticas de seguridad: no correr el contenedor como root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# En producción NO usamos --reload (eso es solo para desarrollo local).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]