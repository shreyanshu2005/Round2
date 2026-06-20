"""
backend/core/cache.py
------------------------
Lightweight Redis response cache for GET endpoints (hotspots, risk — per
Build Guide: TTL=300s, key = f'btip:{endpoint}:{sorted(params)}').

Usage (in a route module)
---------------------------
  from backend.core.cache import cached_response

  @router.get("/hotspots")
  @cached_response(endpoint="hotspots", ttl=300)
  async def get_hotspots(request: Request, limit: int = 10):
      ...

Degrades gracefully to no-op if `app.state.redis` is None (Redis
unavailable) — see backend/main.py startup_redis().
"""

from __future__ import annotations

import functools
import json
import logging

from fastapi import Request

logger = logging.getLogger(__name__)


def _cache_key(endpoint: str, kwargs: dict) -> str:
    params = {k: v for k, v in kwargs.items() if k != "request"}
    sorted_params = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"btip:{endpoint}:{sorted_params}"


def cached_response(endpoint: str, ttl: int = 300):
    """
    Decorator for FastAPI GET route handlers. Requires the decorated
    function to accept a `request: Request` parameter (used to access
    `request.app.state.redis`).
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            redis = getattr(request.app.state, "redis", None)
            key = _cache_key(endpoint, kwargs)

            if redis is not None:
                try:
                    cached = await redis.get(key)
                    if cached is not None:
                        logger.debug("Cache hit: %s", key)
                        return json.loads(cached)
                except Exception as e:
                    logger.warning("Redis GET failed for %s — bypassing cache: %s", key, e)

            result = await func(request, *args, **kwargs)

            if redis is not None:
                try:
                    payload = result.model_dump() if hasattr(result, "model_dump") else result
                    await redis.set(key, json.dumps(payload, default=str), ex=ttl)
                except Exception as e:
                    logger.warning("Redis SET failed for %s — response not cached: %s", key, e)

            return result

        return wrapper

    return decorator