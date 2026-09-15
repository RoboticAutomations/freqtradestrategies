"""API dependencies for authentication and authorization."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db
from src.models.user import User, UserRole
from src.utils.security import TokenData, decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Get the current authenticated user from JWT token.

    Args:
        token: JWT access token from Authorization header.
        db: Database session.

    Returns:
        Authenticated User object.

    Raises:
        HTTPException: If token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    token_data = TokenData.from_payload(payload)
    if token_data is None:
        raise credentials_exception

    if token_data.token_type != "access":
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Verify user is active (placeholder for future deactivation logic).

    Args:
        current_user: Current authenticated user.

    Returns:
        Active user object.
    """
    # Future: Check if user is deactivated
    return current_user


def require_role(allowed_roles: list[UserRole]):
    """Create a dependency that requires specific roles.

    Args:
        allowed_roles: List of roles allowed to access the endpoint.

    Returns:
        Dependency function that validates user role.
    """

    async def role_checker(
        current_user: Annotated[User, Depends(get_current_active_user)],
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return role_checker


# Common role dependencies
RequireAdmin = Depends(require_role([UserRole.ADMIN]))
RequireOperator = Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))
RequireAny = Depends(get_current_active_user)

# Dependency functions for direct use in route parameters
require_admin = require_role([UserRole.ADMIN])
require_operator = require_role([UserRole.ADMIN, UserRole.OPERATOR])

# Type aliases for dependency injection
CurrentUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, RequireAdmin]
OperatorUser = Annotated[User, RequireOperator]
