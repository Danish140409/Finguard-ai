"""
FinGuard AI — Unified FastAPI Application Entrypoint

Initializes the FastAPI application for:
  - Accounts Payable (AP) Anomaly Detection
  - Order-to-Cash (O2C) Credit Management
  - Record-to-Report (R2R) Accounting Controls
"""

from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.deps import get_model_ensemble
from backend.routers import auth, o2c, predict, r2r, reports, transactions
from utils.db import create_all_tables
from utils.logger import get_logger

logger = get_logger("finguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for FastAPI startup and shutdown."""
    logger.info("FinGuard AI API starting up", extra={"environment": settings.ENVIRONMENT})
    
    # Initialize DB tables (creates tables if not existing)
    try:
        create_all_tables()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.warning(f"Could not auto-create tables: {e}")

    # Pre-warm model ensemble if artifacts exist
    try:
        get_model_ensemble()
        logger.info("Ensemble models verified and loaded successfully")
    except Exception as e:
        logger.warning(
            "Model artifacts not found or could not be loaded at startup. "
            "Please run 'python training/train.py' before executing predictions.",
            extra={"error": str(e)},
        )
    
    yield
    
    logger.info("FinGuard AI API shutting down")


app = FastAPI(
    title="FinGuard AI — Unified Enterprise Financial Anomaly Detection API",
    description=(
        "Enterprise-grade REST API covering Accounts Payable (AP), "
        "Order-to-Cash (O2C) Credit Management, and Record-to-Report (R2R) accounting controls. "
        "Provides multi-model ensemble scoring (IF, LOF, OCSVM, XGBoost) and auditable SHAP/LIME explainability."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming request path, method, status code, and latency."""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_sec": round(process_time, 4),
            },
        )
        return response
    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            f"Unhandled exception during {request.method} {request.url.path}: {exc}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_sec": round(process_time, 4),
                "error": str(exc),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred. Please contact system administrator."},
        )


# ---------------------------------------------------------------------------
# Router Ingestion
# ---------------------------------------------------------------------------

api_v1_prefix = "/api/v1"

# AP & Core Routers
app.include_router(auth.router, prefix=api_v1_prefix)
app.include_router(transactions.router, prefix=api_v1_prefix)
app.include_router(predict.router, prefix=api_v1_prefix)
app.include_router(reports.router, prefix=api_v1_prefix)

# O2C & R2R Routers (Mount at /api/v1 and also at /api for flexible client access)
app.include_router(o2c.router, prefix=api_v1_prefix)
app.include_router(o2c.router, prefix="/api")
app.include_router(r2r.router, prefix=api_v1_prefix)
app.include_router(r2r.router, prefix="/api")


@app.get("/", tags=["Health"])
def root():
    """Service landing endpoint."""
    return {
        "service": "FinGuard AI",
        "version": "2.0.0",
        "status": "online",
        "modules": ["Accounts Payable", "Order-to-Cash Credit Management", "Record-to-Report"],
        "docs": "/docs",
        "endpoints": {
            "auth": f"{api_v1_prefix}/auth",
            "transactions": f"{api_v1_prefix}/transactions",
            "predict": f"{api_v1_prefix}/predict",
            "reports": f"{api_v1_prefix}/reports",
            "o2c": "/api/o2c",
            "r2r": "/api/r2r",
            "dashboard_stats": f"{api_v1_prefix}/dashboard/stats",
        },
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe for orchestration and container monitoring."""
    return {"status": "healthy", "service": "finguard-ai-api", "version": "2.0.0"}
