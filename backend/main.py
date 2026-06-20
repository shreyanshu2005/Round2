"""
backend/main.py
------------------
FastAPI entrypoint — Layer 9: wires all REST routers, mounts GraphQL,
adds JWT auth (via per-route Depends, see core/auth.py), Redis response
caching for hotspots/risk, CORS, structured logging, and Prometheus
instrumentation (exposed here, dashboarded in Layer 13).

Routers registered (best-effort import — a missing/incomplete route module
logs a warning and is skipped rather than crashing the whole app, since
not every layer is finished yet per the Current State doc):
  - auth          (/auth/token)            — Layer 9
  - violations    (/api/v1/violations)     — Layer 6
  - hotspots      (/api/v1/hotspots)       — Layer 3
  - risk          (/api/v1/risk)           — Layer 4 (in progress)
  - forecast      (/api/v1/forecast)       — Layer 5
  - recommendations (/api/v1/recommendations) — Layer 7
  - simulation    (/api/v1/simulation)     — Layer 8
  - GraphQL       (/graphql)               — Layer 9
"""

from __future__ import annotations

import logging
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ── Structured logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    stream=sys.stdout,
)
logger = logging.getLogger("btip.main")

app = FastAPI(
    title="BTIP API",
    version="1.0",
    description="Bengaluru Traffic Intelligence Platform — closed-loop decision intelligence API",
)

# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing / structured access log ──────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        '"%s %s" status=%d duration_ms=%.1f',
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Router registration (best-effort, never crashes app startup) ───────────

def _include_router(module_path: str, router_attr: str = "router", prefix: str = "/api/v1"):
    import importlib
    try:
        module = importlib.import_module(module_path)
        router = getattr(module, router_attr)
        app.include_router(router, prefix=prefix)
        logger.info("Registered router: %s (prefix=%s)", module_path, prefix)
    except Exception as e:
        logger.warning(
            "Skipped router %s — not available yet (%s: %s)",
            module_path, type(e).__name__, e,
        )


_include_router("backend.api.routes.auth", prefix="")  # /auth/token, no /api/v1 prefix
_include_router("backend.api.routes.violations")
_include_router("backend.api.routes.hotspots")
_include_router("backend.api.routes.risk")
_include_router("backend.api.routes.forecast")
_include_router("backend.api.routes.recommendations")
_include_router("backend.api.routes.simulation")


# ── GraphQL ───────────────────────────────────────────────────────────────────

try:
    from strawberry.fastapi import GraphQLRouter
    from backend.api.graphql.schema import schema

    graphql_app = GraphQLRouter(schema)
    app.include_router(graphql_app, prefix="/graphql")
    logger.info("GraphQL mounted at /graphql")
except Exception as e:
    logger.warning("GraphQL not mounted (%s: %s)", type(e).__name__, e)


# ── Prometheus instrumentation (exposes /metrics; dashboarded in Layer 13) ──

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus instrumentation enabled at /metrics")
except ImportError:
    logger.info("prometheus-fastapi-instrumentator not installed — /metrics disabled.")


# ── Redis cache (best-effort; degrades to no-cache if Redis unavailable) ────

@app.on_event("startup")
async def startup_redis():
    try:
        import redis.asyncio as aioredis
        import os
        app.state.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )
        await app.state.redis.ping()
        logger.info("Redis cache connected.")
    except Exception as e:
        app.state.redis = None
        logger.warning("Redis unavailable — caching disabled (%s: %s)", type(e).__name__, e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)