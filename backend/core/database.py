"""
backend/core/database.py

SQLAlchemy engine + session factory for PostgreSQL/PostGIS/TimescaleDB.
Sync engine used for now (ingestion scripts); Layer 9 adds an async engine
for the FastAPI app.
"""
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_engine():
    """Callable accessor for the module-level engine — some callers (e.g.
    GraphQL schema setup, scripts that import lazily to avoid connecting
    at import time) expect a function rather than a bare variable."""
    return engine


@contextmanager
def get_session():
    """Context-managed DB session: `with get_session() as session: ...`"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def healthcheck() -> bool:
    """Returns True if a trivial query succeeds against the DB."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False