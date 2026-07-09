from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# We use a simple token scheme. The URL matches our new auth router login endpoint.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# Simple in-memory session store: mapping token -> user dictionary
ACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}

def create_session_token(user: dict[str, Any]) -> str:
    """Create a simple token (UUID) and store the user session in memory."""
    token = str(uuid.uuid4())
    ACTIVE_SESSIONS[token] = user.copy()
    return token

async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict[str, Any]:
    """
    FastAPI dependency that extracts the bearer token and looks it up in memory.
    If no token is provided or it's invalid, it returns a default demo user.
    """
    if not token or token not in ACTIVE_SESSIONS:
        # Fallback to demo user for simplicity as per requirements
        return {"sub": "demo-analyst", "name": "Demo Analyst", "role": "fraud_analyst"}
    return ACTIVE_SESSIONS[token]

async def require_analyst(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    """Dependency that ensures the user has sufficient permissions."""
    allowed_roles = {"fraud_analyst", "compliance_officer", "admin"}
    if user.get("role") not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return user
