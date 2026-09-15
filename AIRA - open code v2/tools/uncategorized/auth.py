"""Authentication API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db
from src.models.user import User, UserRole
from src.api.deps import get_current_active_user
from src.utils.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from src.config import settings

router = APIRouter()


class TokenResponse(BaseModel):
    """Token response schema."""

    status: str = "success"
    data: dict


class UserResponse(BaseModel):
    """User info response schema."""

    id: str
    username: str
    role: str
    preferences: dict


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""

    refresh_token: str


@router.post("/token", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Authenticate user and return access token.

    Args:
        form_data: OAuth2 form with username and password.
        db: Database session.

    Returns:
        Token response with access and refresh tokens.

    Raises:
        HTTPException: If credentials are invalid.
    """
    # Find user by username
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    token_data = {
        "sub": user.username,
        "user_id": user.id,
        "role": user.role.value,
    }
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        data={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.auth.token_expire_minutes * 60,
        }
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Refresh access token using refresh token.

    Args:
        refresh_token: Valid refresh token.
        db: Database session.

    Returns:
        New token response.

    Raises:
        HTTPException: If refresh token is invalid.
    """
    payload = decode_token(request.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = payload.get("user_id")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # Create new access token
    token_data = {
        "sub": user.username,
        "user_id": user.id,
        "role": user.role.value,
    }
    new_access_token = create_access_token(token_data)

    return TokenResponse(
        data={
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": settings.auth.token_expire_minutes * 60,
        }
    )


@router.get("/me")
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> dict:
    """Get current authenticated user information.

    This endpoint is protected and requires a valid access token.

    Returns:
        Current user information.
    """
    return {
        "status": "success",
        "data": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role.value,
            "preferences": current_user.preferences,
        },
    }
