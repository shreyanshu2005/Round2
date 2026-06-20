"""
backend/tests/test_api.py
----------------------------
Layer 9 API tests:
  - Auth flow: /auth/token issues valid JWT; protected routes 401 without
    token, 200/403 with token depending on role
  - Health check
  - GraphQL endpoint mounts and responds to a basic query
  - Response shape sanity on a protected demo route

Run with: pytest backend/tests/test_api.py -v
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.core.auth import Role, require_role, get_current_user, TokenPayload


@pytest.fixture
def app() -> FastAPI:
    """
    Minimal app for testing auth wiring in isolation — avoids depending on
    every backend route module (some are still in progress per the
    Current State doc) being importable.
    """
    from backend.api.routes.auth import router as auth_router

    test_app = FastAPI()
    test_app.include_router(auth_router)

    @test_app.get("/protected/any")
    async def protected_any(user: TokenPayload = Depends(get_current_user)):
        return {"user": user.sub, "role": user.role.value}

    @test_app.get("/protected/commander-only")
    async def protected_commander(user: TokenPayload = Depends(require_role("Commander"))):
        return {"user": user.sub, "role": user.role.value}

    @test_app.get("/health")
    async def health():
        return {"status": "ok"}

    return test_app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Auth flow ─────────────────────────────────────────────────────────────────

class TestAuthFlow:
    def test_token_issued_for_valid_credentials(self, client):
        resp = client.post(
            "/auth/token",
            data={"username": "commander1", "password": "demo-pass-commander"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_token_rejected_for_invalid_credentials(self, client):
        resp = client.post(
            "/auth/token",
            data={"username": "commander1", "password": "wrong-password"},
        )
        assert resp.status_code == 401

    def test_token_rejected_for_unknown_user(self, client):
        resp = client.post(
            "/auth/token",
            data={"username": "nobody", "password": "irrelevant"},
        )
        assert resp.status_code == 401

    def test_protected_route_401_without_token(self, client):
        resp = client.get("/protected/any")
        assert resp.status_code == 401

    def test_protected_route_200_with_valid_token(self, client):
        token_resp = client.post(
            "/auth/token",
            data={"username": "analyst1", "password": "demo-pass-analyst"},
        )
        token = token_resp.json()["access_token"]
        resp = client.get("/protected/any", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "Analyst"

    def test_protected_route_401_with_garbage_token(self, client):
        resp = client.get("/protected/any", headers={"Authorization": "Bearer not-a-real-token"})
        assert resp.status_code == 401

    def test_role_based_access_denies_wrong_role(self, client):
        token_resp = client.post(
            "/auth/token",
            data={"username": "officer1", "password": "demo-pass-officer"},
        )
        token = token_resp.json()["access_token"]
        resp = client.get(
            "/protected/commander-only", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 403

    def test_role_based_access_allows_correct_role(self, client):
        token_resp = client.post(
            "/auth/token",
            data={"username": "commander1", "password": "demo-pass-commander"},
        )
        token = token_resp.json()["access_token"]
        resp = client.get(
            "/protected/commander-only", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200


# ── Token internals ──────────────────────────────────────────────────────────

class TestTokenInternals:
    def test_create_and_verify_token_roundtrip(self):
        from backend.core.auth import create_token, verify_token

        token = create_token("test_user", Role.OFFICER)
        payload = verify_token(token)
        assert payload.sub == "test_user"
        assert payload.role == Role.OFFICER

    def test_verify_rejects_tampered_token(self):
        from backend.core.auth import create_token, verify_token
        from fastapi import HTTPException

        token = create_token("test_user", Role.OFFICER)
        tampered = token[:-3] + "xyz"
        with pytest.raises(HTTPException) as exc_info:
            verify_token(tampered)
        assert exc_info.value.status_code == 401


# ── GraphQL mount (best-effort — skipped if strawberry not installed) ───────

class TestGraphQL:
    def test_graphql_schema_imports_and_has_query_type(self):
        strawberry = pytest.importorskip("strawberry")
        from backend.api.graphql.schema import schema

        assert schema is not None
        # Basic introspection query should succeed without raising
        result = schema.execute_sync("{ __typename }")
        assert result.errors is None or len(result.errors) == 0

    def test_graphql_mounts_on_app(self):
        pytest.importorskip("strawberry")
        from strawberry.fastapi import GraphQLRouter
        from backend.api.graphql.schema import schema

        test_app = FastAPI()
        test_app.include_router(GraphQLRouter(schema), prefix="/graphql")
        gql_client = TestClient(test_app)

        resp = gql_client.post("/graphql", json={"query": "{ __typename }"})
        assert resp.status_code == 200