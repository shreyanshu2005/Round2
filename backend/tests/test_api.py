"""
backend/tests/test_api.py
Layer 9 — API test suite. Uses httpx.AsyncClient against the FastAPI app.

Covers:
  - 401 on protected routes without token, 200 with valid token
  - /auth/token login flow (valid + invalid credentials)
  - Response schema spot-checks for each main route
  - Pagination on /violations
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.main import app

BASE = "/api/v1"


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def commander_token(client: AsyncClient) -> str:
    resp = await client.post(
        "/auth/token",
        data={"username": "commander1", "password": "commander123"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Auth flow
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_login_success(client: AsyncClient):
    resp = await client.post(
        "/auth/token",
        data={"username": "commander1", "password": "commander123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["role"] == "Commander"


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient):
    resp = await client.post(
        "/auth/token",
        data={"username": "commander1", "password": "wrongpass"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_without_token(client: AsyncClient):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_with_token(client: AsyncClient, commander_token: str):
    resp = await client.get(
        "/auth/me", headers={"Authorization": f"Bearer {commander_token}"}
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "Commander"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Endpoint schema spot-checks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hotspots_schema(client: AsyncClient):
    resp = await client.get(f"{BASE}/hotspots")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if body:
        item = body[0]
        for key in ("cluster_id", "centroid_lat", "centroid_lng", "persistence_score"):
            assert key in item


@pytest.mark.asyncio
async def test_risk_schema(client: AsyncClient):
    resp = await client.get(
        f"{BASE}/risk", params={"zone_id": "1", "shift": "Evening", "date": "2024-01-15"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["risk_score"] <= 100
    assert len(body["shap_explanations"]) == 5


@pytest.mark.asyncio
async def test_forecast_schema(client: AsyncClient):
    resp = await client.get(f"{BASE}/forecast", params={"junction_id": "5", "horizon": "24h"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if body:
        point = body[0]
        assert point["p10"] <= point["p50"] <= point["p90"]


@pytest.mark.asyncio
async def test_violations_pagination(client: AsyncClient):
    resp = await client.get(f"{BASE}/violations", params={"limit": 10, "offset": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) <= 10


@pytest.mark.asyncio
async def test_recommendations_requires_total_officers(client: AsyncClient):
    resp = await client.get(
        f"{BASE}/recommendations",
        params={"shift": "Evening", "date": "2024-01-15", "total_officers": 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if body:
        assert "shap_explanations" in body[0]


@pytest.mark.asyncio
async def test_simulation_post(client: AsyncClient):
    payload = {
        "zone_allocations": [{"zone_id": "BTP051", "n_officers": 3}],
        "shift": "Evening",
        "date": "2024-01-15",
    }
    resp = await client.post(f"{BASE}/simulation", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence_band"]["p10"] <= body["confidence_band"]["p50"] <= body["confidence_band"]["p90"]