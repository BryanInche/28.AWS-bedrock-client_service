"""
Conexión a Redis. Requiere Redis Stack (no el Redis básico) porque
usamos búsqueda vectorial (módulo RediSearch), necesaria para el
cache semántico.

    docker run -d --name redis-cache -p 6379:6379 redis/redis-stack-server:latest
"""

import os

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


def get_redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)