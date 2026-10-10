"""
Cache semántico sobre Redis: antes de llamar al LLM, busca si ya se
respondió algo MUY PARECIDO en significado (no texto exacto). Si hay
coincidencia, devuelve la respuesta guardada — cero costo de tokens.
Si no hay coincidencia, llama al LLM y GUARDA la respuesta para la
próxima vez.

Requiere Redis Stack (módulo RediSearch) para búsqueda vectorial.
"""

import hashlib
import time

import numpy as np
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from redis_client import get_redis_client

# Mismo modelo de embeddings que ya usa el RAG (backend/rag/embeddings.py)
# — consistencia, y ya está en requirements.txt del backend.
MODELO_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONES = 384

NOMBRE_INDICE = "cache_idx"
PREFIJO_CLAVES = "cache:"

# Umbral de similitud: qué tan "parecidas" deben ser 2 preguntas para
# considerarlas la MISMA consulta. Con distancia coseno, 0 = idénticas,
# 2 = opuestas. 0.15 es un punto de partida razonable — más bajo =
# más estricto (menos "cache hits", pero más seguro).
UMBRAL_DISTANCIA = 0.15

TTL_SEGUNDOS = 60 * 60 * 24 * 7  # 7 días de vida por entrada de cache


class SemanticCache:
    def __init__(self):
        self.redis = get_redis_client()
        self.modelo = SentenceTransformer(MODELO_EMBEDDING)
        self._crear_indice_si_no_existe()

    def _crear_indice_si_no_existe(self):
        try:
            self.redis.ft(NOMBRE_INDICE).info()
            return  # ya existe, no hacer nada
        except Exception:
            pass  # no existe, lo creamos abajo

        esquema = (
            TextField("prompt"),
            TextField("response"),
            VectorField(
                "vector",
                "FLAT",  # búsqueda exacta (suficiente para volúmenes chicos/medianos)
                {"TYPE": "FLOAT32", "DIM": DIMENSIONES, "DISTANCE_METRIC": "COSINE"},
            ),
        )
        self.redis.ft(NOMBRE_INDICE).create_index(
            esquema,
            definition=IndexDefinition(prefix=[PREFIJO_CLAVES], index_type=IndexType.HASH),
        )
        print(f"[SemanticCache] Índice '{NOMBRE_INDICE}' creado en Redis.")

    def _embed(self, texto: str) -> bytes:
        vector = self.modelo.encode(texto, normalize_embeddings=True)
        return vector.astype(np.float32).tobytes()

    def get(self, prompt: str) -> str | None:
        """Busca una respuesta ya guardada, semánticamente parecida a 'prompt'."""
        vector_bytes = self._embed(prompt)

        consulta = (
            Query("(*)=>[KNN 1 @vector $vec AS distancia]")
            .sort_by("distancia")
            .return_fields("response", "distancia")
            .dialect(2)
        )

        resultados = self.redis.ft(NOMBRE_INDICE).search(
            consulta, query_params={"vec": vector_bytes}
        )

        if not resultados.docs:
            return None

        mejor = resultados.docs[0]
        distancia = float(mejor.distancia)

        if distancia <= UMBRAL_DISTANCIA:
            print(f"[SemanticCache] HIT (distancia={distancia:.4f}) — se ahorró una llamada al LLM.")
            return mejor.response

        print(f"[SemanticCache] MISS (la más cercana tenía distancia={distancia:.4f}, umbral={UMBRAL_DISTANCIA}).")
        return None

    def set(self, prompt: str, response: str) -> None:
        """Guarda una nueva pregunta+respuesta en el cache."""
        clave = PREFIJO_CLAVES + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        vector_bytes = self._embed(prompt)

        self.redis.hset(clave, mapping={"prompt": prompt, "response": response, "vector": vector_bytes})
        self.redis.expire(clave, TTL_SEGUNDOS)