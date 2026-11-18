"""
FinGuard AI — Auth Router

Endpoints:
  POST /auth/login   — issue JWT token
  POST /auth/logout  — revoke JWT token
  GET  /auth/me      — current user info
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.deps import (
    blacklist_token,
    create_access_token,
    get_current_user,
    hash_password,
    oauth2_scheme,
    verify_password,
)
import uuid
from backend.config import settings
from backend.schemas import LoginSchema, TokenSchema, UserRegisterSchema, UserSchema
from utils.db import User, get_db
from utils.logger import get_logger, log_audit_event

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenSchema, summary="Obtain a JWT access token")
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenSchema:
    """
    Authenticate with username + password.
    Returns a JWT bearer token valid for ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        log_audit_event(
            action="LOGIN_FAILED",
            entity_type="user",
            entity_id=None,
            performed_by=form_data.username,
            details={"reason": "invalid_credentials"},
            db_session=db,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled")

    expire_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token        = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=expire_delta,
    )
    log_audit_event(
        action="LOGIN_SUCCESS",
        entity_type="user",
        entity_id=user.user_id,
        performed_by=user.username,
        db_session=db,
    )
    return TokenSchema(
        access_token=token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout", summary="Revoke the current JWT token")
def logout(
    token:        Annotated[str, Depends(oauth2_scheme)],
    current_user: Annotated[User, Depends(get_current_user)],
    db:           Annotated[Session, Depends(get_db)],
) -> dict:
    """Add the current token to the blacklist."""
    blacklist_token(token)
    log_audit_event(
        action="LOGOUT",
        entity_type="user",
        entity_id=current_user.user_id,
        performed_by=current_user.username,
        db_session=db,
    )
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserSchema, summary="Get current user info")
def get_me(current_user: Annotated[User, Depends(get_current_user)]) -> UserSchema:
    """Return the authenticated user's profile."""
    return current_user


@router.post("/register", response_model=UserSchema, summary="Register a new platform user")
def register(
    payload: UserRegisterSchema,
    db: Annotated[Session, Depends(get_db)],
) -> UserSchema:
    """Register a new user account for the platform."""
    existing = db.query(User).filter(
        (User.username == payload.username) | (User.email == payload.email)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email is already registered."
        )
    new_user = User(
        user_id=uuid.uuid4(),
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    log_audit_event(
        action="USER_REGISTERED",
        entity_type="user",
        entity_id=new_user.user_id,
        performed_by=new_user.username,
        details={"role": new_user.role, "email": new_user.email},
        db_session=db,
    )
    return new_user
