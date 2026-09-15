"""Security utilities for password hashing and JWT token management."""

from datetime import datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.config import settings
from src.models.user import UserRole

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password.

    Returns:
        Hashed password string.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    Args:
        plain_password: Plain text password to verify.
        hashed_password: Stored password hash.

    Returns:
        True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token.
        expires_delta: Optional expiration time delta.

    Returns:
        Encoded JWT token string.
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.auth.token_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.auth.jwt_secret,
        algorithm=settings.auth.jwt_algorithm,
    )
    return encoded_jwt


def create_refresh_token(data: dict[str, Any]) -> str:
    """Create a JWT refresh token with longer expiration.

    Args:
        data: Data to encode in the token.

    Returns:
        Encoded JWT refresh token string.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.auth.refresh_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.auth.jwt_secret,
        algorithm=settings.auth.jwt_algorithm,
    )
    return encoded_jwt


def decode_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT token.

    Args:
        token: JWT token string.

    Returns:
        Decoded token payload or None if invalid.
    """
    try:
        payload = jwt.decode(
            token,
            settings.auth.jwt_secret,
            algorithms=[settings.auth.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


class TokenData:
    """Data extracted from a JWT token."""

    def __init__(
        self,
        username: str,
        user_id: str,
        role: UserRole,
        exp: datetime,
        token_type: str = "access",
    ):
        self.username = username
        self.user_id = user_id
        self.role = role
        self.exp = exp
        self.token_type = token_type

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TokenData | None":
        """Create TokenData from a decoded JWT payload.

        Args:
            payload: Decoded JWT payload dict.

        Returns:
            TokenData instance or None if invalid.
        """
        try:
            username = payload.get("sub")
            user_id = payload.get("user_id")
            role_str = payload.get("role")
            exp = payload.get("exp")
            token_type = payload.get("type", "access")

            if not all([username, user_id, role_str, exp]):
                return None

            role = UserRole(role_str)
            exp_dt = datetime.fromtimestamp(exp)

            return cls(
                username=username,
                user_id=user_id,
                role=role,
                exp=exp_dt,
                token_type=token_type,
            )
        except (ValueError, KeyError):
            return None
