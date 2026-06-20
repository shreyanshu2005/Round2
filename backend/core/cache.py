"""
backend/core/cache.py
Redis caching helpers for hot GET endpoints (/hotspots, /risk).
Uses aioredis-compatible redis.asyncio client. TTL default 300s (5 min).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as aioredis

from backend.core.config import settings

_redis_client: Optional[aioredis.Redis] = None

DEFAULT_TTL_SECONDS = 300


def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
    return _redis_client


def make_cache_key(endpoint: str, params: dict) -> str:
    """btip:{endpoint}:{sorted(params)} — stable hash of query params."""
    sorted_params = json.dumps(sorted(params.items()), default=str)
    digest = hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
    return f"btip:{endpoint}:{digest}"


async def cache_get(key: str) -> Optional[Any]:
    client = get_redis()
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        # Cache is best-effort — never break the request path on Redis failure
        return None


async def cache_set(key: str, value: Any, ttl: int = DEFAULT_TTL_SECONDS) -> None:
    client = get_redis()
    try:
        await client.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception:
        pass


async def cached_endpoint(endpoint: str, params: dict, fetch_fn, ttl: int = DEFAULT_TTL_SECONDS):
    """
    Generic cache-aside wrapper.
    fetch_fn: zero-arg async callable that produces the response on a cache miss.
    """
    key = make_cache_key(endpoint, params)
    hit = await cache_get(key)
    if hit is not None:
        return hit
    result = await fetch_fn()
    await cache_set(key, result, ttl=ttl)
    return result