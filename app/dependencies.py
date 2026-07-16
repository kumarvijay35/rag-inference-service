"""Shared FastAPI dependencies.

Service-to-service auth: this service sits behind Django and must never be
called directly by browsers. Django attaches a shared secret in the
X-Internal-Api-Key header; anything without it gets a 401.

Implemented as a FastAPI dependency -> one line to protect any router.
"""

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings


async def verify_internal_key(
    x_internal_api_key: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> None:
    if x_internal_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key",
        )


def get_state(request: Request):
    """Access objects created once at startup (embedding model, clients)."""
    return request.app.state
