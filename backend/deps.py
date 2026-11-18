"""
FinGuard AI — FastAPI Dependencies

Provides reusable dependencies for:
  - Database session management
  - JWT authentication
  - Role-based access control
  - Model ensemble loading (cached)
"""

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import joblib
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.config import settings
from utils.db import AuditLog, User, get_db
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

import bcrypt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        try:
            return pwd_context.verify(plain, hashed)
        except Exception:
            return False


def hash_password(plain: str) -> str:
    try:
        return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except Exception:
        return pwd_context.hash(plain)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# In-memory token blacklist (use Redis in production)
_token_blacklist: set[str] = set()


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire    = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def blacklist_token(token: str) -> None:
    _token_blacklist.add(token)


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db:    Annotated[Session, Depends(get_db)],
) -> User:
    """Validate JWT and return the current User ORM object."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token in _token_blacklist:
        raise HTTPException(status_code=401, detail="Token has been revoked")

    try:
        payload  = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exc
    return user


def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Raise 403 if the current user is not an admin."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation",
        )
    return current_user


# ---------------------------------------------------------------------------
# Model ensemble loading (cached singleton)
# ---------------------------------------------------------------------------

_models_cache: dict | None = None
_scaler_cache = None
_metadata_cache: dict | None = None


def get_model_ensemble() -> tuple[dict, Any, dict]:
    """
    Load and cache the production model ensemble.
    Returns (models_dict, scaler, metadata).
    Raises HTTP 503 if models are not found.
    """
    global _models_cache, _scaler_cache, _metadata_cache

    if _models_cache is not None:
        return _models_cache, _scaler_cache, _metadata_cache

    prod_dir = Path(settings.MODEL_PATH)
    required = [
        "isolation_forest.joblib",
        "lof.joblib",
        "ocsvm.joblib",
        "xgboost.joblib",
        "scaler.joblib",
        "metadata.json",
    ]
    missing = [f for f in required if not (prod_dir / f).exists()]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model artifacts not found: {missing}. "
                f"Run 'python training/train.py' first."
            ),
        )

    try:
        _models_cache = {
            "isolation_forest": joblib.load(prod_dir / "isolation_forest.joblib"),
            "lof":              joblib.load(prod_dir / "lof.joblib"),
            "ocsvm":            joblib.load(prod_dir / "ocsvm.joblib"),
            "xgboost":          joblib.load(prod_dir / "xgboost.joblib"),
        }
        _scaler_cache    = joblib.load(prod_dir / "scaler.joblib")
        with open(prod_dir / "metadata.json") as f:
            _metadata_cache = json.load(f)

        logger.info("Model ensemble loaded", extra={"version": _metadata_cache.get("version")})
        return _models_cache, _scaler_cache, _metadata_cache
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load model ensemble: {str(e)}",
        )


def invalidate_model_cache() -> None:
    """Call after retraining to force reload on next request."""
    global _models_cache, _scaler_cache, _metadata_cache
    _models_cache = _scaler_cache = _metadata_cache = None
    logger.info("Model cache invalidated")
