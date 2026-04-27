"""JWT authentication for the FastAPI layer.

Every API request that modifies state (inference, FHIR write) requires
a valid JWT bearer token. This lets the audit log record WHO triggered
each event — a core 21 CFR Part 11 requirement.

For the portfolio demo: a single hardcoded demo user can obtain a token
via POST /auth/token. Production would replace this with an identity
provider (Azure AD, Okta, etc.).

Why JWT over API keys:
  - Carries the actor identity (subject claim) in the token itself.
  - Stateless — no session store needed.
  - Standard in healthcare APIs (SMART on FHIR uses OAuth2 + JWT).
  - Expiry is built in — tokens auto-invalidate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from api.core.config import Settings, get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Demo users — in production, replace with a real user store / IdP
DEMO_USERS: dict[str, str] = {
    "demo": "demo123",
    "radiologist": "rad456",
}


class TokenData(BaseModel):
    """Decoded JWT payload."""
    username: str


class Token(BaseModel):
    """OAuth2 token response."""
    access_token: str
    token_type: str


def create_access_token(
    subject: str,
    settings: Settings,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        subject: The user identity to embed (e.g. username).
        settings: App settings (for secret key and algorithm).
        expires_delta: Optional custom expiry. Defaults to settings value.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire}
    return cast(str, jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm))


def authenticate_user(username: str, password: str) -> str | None:
    """Verify username/password and return the username if valid, else None."""
    if DEMO_USERS.get(username) == password:
        return username
    return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency: decode JWT and return the authenticated username.

    Raises:
        HTTPException 401: If token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError as err:
        raise credentials_exception from err
