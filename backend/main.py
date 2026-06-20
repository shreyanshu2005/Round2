"""
backend/main.py
BTIP FastAPI entrypoint — Layer 9 (final backend wiring).

Wires:
  - All REST routers (violations, hotspots, risk, forecast, recommendations, simulation)
  - /auth/token login route + JWT middleware via per-route Depends
  - Strawberry GraphQL mounted at /graphql
  - Redis cache (applied inside hotspots.py / risk.py route handlers)
  - Structured logging
  - Prometheus instrumentation (kept ready for Layer 13)
"""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from strawberry.fastapi import GraphQLRouter

from backend.api.graphql.schema import schema
from backend.api.routes import (
    forecast,
    hotspots,
    recommendations,
    risk,
    simulation,
    violations,
)
from backend.core.auth import Token, get_current_user, login_for_access_token

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("btip")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(title="BTIP API", version="1.0", description="Bengaluru Traffic Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth route
# ---------------------------------------------------------------------------
@app.post("/auth/token", response_model=Token, tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Demo login — accepts username/password, returns a 24h JWT with role claim."""
    return login_for_access_token(form_data)


@app.get("/auth/me", tags=["auth"])
async def whoami(current_user=Depends(get_current_user)):
    return {"username": current_user.username, "role": current_user.role}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# REST routers — all prefixed /api/v1
# ---------------------------------------------------------------------------
API_PREFIX = "/api/v1"

app.include_router(violations.router, prefix=API_PREFIX, tags=["violations"])
app.include_router(hotspots.router, prefix=API_PREFIX, tags=["hotspots"])
app.include_router(risk.router, prefix=API_PREFIX, tags=["risk"])
app.include_router(forecast.router, prefix=API_PREFIX, tags=["forecast"])
app.include_router(recommendations.router, prefix=API_PREFIX, tags=["recommendations"])
app.include_router(simulation.router, prefix=API_PREFIX, tags=["simulation"])

# ---------------------------------------------------------------------------
# GraphQL — mounted at /graphql, mirrors REST resolvers
# ---------------------------------------------------------------------------
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


@app.on_event("startup")
async def on_startup():
    logger.info("BTIP API starting up — REST (%s) + GraphQL (/graphql) live", API_PREFIX)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("BTIP API shutting down")