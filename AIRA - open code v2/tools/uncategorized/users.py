"""User management API routes (Admin only)."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db
from src.models.user import User, UserRole
from src.api.deps import get_current_active_user, require_admin
from src.utils.security import hash_password

router = APIRouter()


class UserCreate(BaseModel):
    """Schema for creating a new user."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field(default="readonly")


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    username: Optional[str] = Field(None, min_length=3, max_length=50)
    password: Optional[str] = Field(None, min_length=6)
    role: Optional[str] = None


class UserPreferencesUpdate(BaseModel):
    """Schema for updating user preferences."""

    theme: Optional[str] = None
    notifications_enabled: Optional[bool] = None
    default_timeframe: Optional[str] = None
    dashboard_layout: Optional[dict] = None
    refresh_interval: Optional[int] = None
    default_view: Optional[str] = None
    alert_critical: Optional[bool] = None
    alert_warning: Optional[bool] = None
    alert_info: Optional[bool] = None


class UserOut(BaseModel):
    """User output schema."""

    id: str
    username: str
    role: str
    preferences: dict
    created_at: str


@router.get("")
async def list_users(
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
    skip: int = 0,
    limit: int = 50,
) -> dict:
    """List all users (Admin only).

    Args:
        db: Database session.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of users.
    """
    # Get total count
    count_result = await db.execute(select(func.count(User.id)))
    total = count_result.scalar()

    # Get users
    result = await db.execute(
        select(User).offset(skip).limit(limit).order_by(User.created_at.desc())
    )
    users = result.scalars().all()

    return {
        "status": "success",
        "data": {
            "users": [
                {
                    "id": u.id,
                    "username": u.username,
                    "role": u.role.value,
                    "preferences": u.preferences,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                }
                for u in users
            ],
            "total": total,
            "skip": skip,
            "limit": limit,
        },
    }


@router.post("")
async def create_user(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
) -> dict:
    """Create a new user (Admin only).

    Args:
        user_data: User creation data.
        db: Database session.

    Returns:
        Created user data.
    """
    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    # Validate role
    try:
        role = UserRole(user_data.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {[r.value for r in UserRole]}",
        )

    # Create user
    user = User(
        username=user_data.username,
        password_hash=hash_password(user_data.password),
        role=role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return {
        "status": "success",
        "message": "User created successfully",
        "data": {
            "id": user.id,
            "username": user.username,
            "role": user.role.value,
        },
    }


@router.get("/{user_id}")
async def get_user(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
) -> dict:
    """Get a specific user (Admin only).

    Args:
        user_id: User UUID.
        db: Database session.

    Returns:
        User data.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return {
        "status": "success",
        "data": {
            "id": user.id,
            "username": user.username,
            "role": user.role.value,
            "preferences": user.preferences,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        },
    }


@router.patch("/{user_id}")
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> dict:
    """Update a user (Admin only).

    Args:
        user_id: User UUID.
        user_data: User update data.
        db: Database session.
        current_user: Current admin user.

    Returns:
        Updated user data.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent admin from changing their own role
    if user.id == current_user.id and user_data.role and user_data.role != current_user.role.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    # Update fields
    if user_data.username:
        # Check username uniqueness
        existing = await db.execute(
            select(User).where(User.username == user_data.username, User.id != user_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )
        user.username = user_data.username

    if user_data.password:
        user.password_hash = hash_password(user_data.password)

    if user_data.role:
        try:
            user.role = UserRole(user_data.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {[r.value for r in UserRole]}",
            )

    await db.commit()
    await db.refresh(user)

    return {
        "status": "success",
        "message": "User updated successfully",
        "data": {
            "id": user.id,
            "username": user.username,
            "role": user.role.value,
        },
    }


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> dict:
    """Delete a user (Admin only).

    Args:
        user_id: User UUID.
        db: Database session.
        current_user: Current admin user.

    Returns:
        Success message.
    """
    # Prevent self-deletion
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    await db.delete(user)
    await db.commit()

    return {
        "status": "success",
        "message": "User deleted successfully",
    }


# User preferences (self-update)
@router.patch("/me/preferences")
async def update_my_preferences(
    preferences: UserPreferencesUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> dict:
    """Update current user's preferences.

    Args:
        preferences: Preferences to update.
        db: Database session.
        current_user: Current authenticated user.

    Returns:
        Updated preferences.
    """
    # Merge with existing preferences
    current_prefs = current_user.preferences or {}

    if preferences.theme is not None:
        current_prefs["theme"] = preferences.theme
    if preferences.notifications_enabled is not None:
        current_prefs["notifications_enabled"] = preferences.notifications_enabled
    if preferences.default_timeframe is not None:
        current_prefs["default_timeframe"] = preferences.default_timeframe
    if preferences.dashboard_layout is not None:
        current_prefs["dashboard_layout"] = preferences.dashboard_layout
    if preferences.refresh_interval is not None:
        current_prefs["refresh_interval"] = preferences.refresh_interval
    if preferences.default_view is not None:
        current_prefs["default_view"] = preferences.default_view
    if preferences.alert_critical is not None:
        current_prefs["alert_critical"] = preferences.alert_critical
    if preferences.alert_warning is not None:
        current_prefs["alert_warning"] = preferences.alert_warning
    if preferences.alert_info is not None:
        current_prefs["alert_info"] = preferences.alert_info

    current_user.preferences = current_prefs
    await db.commit()
    await db.refresh(current_user)

    return {
        "status": "success",
        "message": "Preferences updated",
        "data": {
            "preferences": current_user.preferences,
        },
    }


@router.patch("/me/password")
async def change_my_password(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_active_user)],
    current_password: str = "",
    new_password: str = "",
) -> dict:
    """Change current user's password.

    Args:
        db: Database session.
        current_user: Current authenticated user.
        current_password: Current password for verification.
        new_password: New password.

    Returns:
        Success message.
    """
    from src.utils.security import verify_password

    if not verify_password(current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    if len(new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters",
        )

    current_user.password_hash = hash_password(new_password)
    await db.commit()

    return {
        "status": "success",
        "message": "Password changed successfully",
    }
