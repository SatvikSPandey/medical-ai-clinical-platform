"""Auth router - POST /auth/token."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.core.config import Settings, get_settings
from api.core.security import authenticate_user, create_access_token
from api.models.schemas import TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Obtain a JWT bearer token. Use in Authorization: Bearer <token> header."""
    username = authenticate_user(form.username, form.password)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(username, settings)
    return TokenResponse(access_token=token, token_type="bearer")
