"""
backend/core/auth.py
JWT authentication + role-based access control for BTIP.
Roles: Commander, Officer, Analyst
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from backend.core.config import settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Demo users — hardcoded for hackathon demo only. Replace with DB-backed
# users before any real deployment.
DEMO_USERS = {
    "commander1": {
        "username": "commander1",
        "hashed_password": pwd_context.hash("commander123"),
        "role": "Commander",
    },
    "officer1": {
        "username": "officer1",
        "hashed_password": pwd_context.hash("officer123"),
        "role": "Officer",
    },
    "analyst1": {
        "username": "analyst1",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "Analyst",
    },
}

VALID_ROLES = {"Commander", "Officer", "Analyst"}


class TokenData(BaseModel):
    username: str
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_hours: int = ACCESS_TOKEN_EXPIRE_HOURS


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = DEMO_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_token(username: str, role: str) -> str:
    """Create a signed JWT containing username + role claims, 24h expiry."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}")
    expire = dt.datetime.utcnow() + dt.timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> TokenData:
    """Decode + validate a JWT. Raises HTTPException(401) on failure."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        return TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    return verify_token(token)


def require_role(*allowed_roles: str):
    """
    Dependency factory for role-gated routes.
    Usage: @router.get("/x", dependencies=[Depends(require_role("Commander"))])
    """

    async def role_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' not permitted. Requires one of: {allowed_roles}",
            )
        return current_user

    return role_checker


# ---------------------------------------------------------------------------
# /auth/token route logic (wired into main.py)
# ---------------------------------------------------------------------------
def login_for_access_token(form_data: OAuth2PasswordRequestForm) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_token(username=user["username"], role=user["role"])
    return Token(access_token=token, role=user["role"])