from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.core.security import create_session_token

router = APIRouter()

# Simple in-memory user store for simplified auth logic
# Format: { "username": {"password": "pwd", "name": "Name", "role": "fraud_analyst"} }
USERS_DB = {}

class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str | None = None

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/register")
async def register(body: RegisterRequest) -> dict[str, str]:
    """Create a simple account."""
    if body.username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists",
        )
    
    # Store plain-text password and simple details per requirements (no advanced crypto)
    USERS_DB[body.username] = {
        "password": body.password,
        "name": body.name or body.username,
        "role": "fraud_analyst",
    }
    return {"message": "User registered successfully"}

@router.post("/login")
async def login(body: LoginRequest) -> dict[str, str]:
    """Login and receive a simple bearer token."""
    user = USERS_DB.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    
    # Create simple uuid token and store it
    token = create_session_token({
        "sub": body.username,
        "name": user["name"],
        "role": user["role"],
    })
    
    return {"access_token": token, "token_type": "bearer"}
