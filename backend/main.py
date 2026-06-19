"""
BTIP FastAPI application entrypoint.

Start with:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.hotspots import router as hotspots_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BTIP API starting up …")
    yield
    logger.info("BTIP API shutting down …")


app = FastAPI(
    title="BTIP — Bengaluru Traffic Intelligence Platform",
    version="1.0.0",
    description="AI Command Center for Traffic Enforcement — Flipkart Gridlock 2.0",
    lifespan=lifespan,
)

# ── CORS — allow frontend dev server ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(hotspots_router, prefix="/api/v1", tags=["Hotspots"])


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "btip-api", "version": "1.0.0"}


@app.get("/", tags=["Health"])
def root():
    return {"message": "BTIP API — see /docs for endpoints"}