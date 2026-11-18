"""
FinGuard AI — API Integration Tests

Validates core FastAPI endpoints:
  - GET / (Root health check)
  - GET /health
  - Unauthenticated rejection on protected routes
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify API root status and service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "FinGuard AI"
    assert data["status"] == "online"


def test_health_endpoint():
    """Verify health probe returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "finguard-ai-api"


def test_unauthenticated_transactions_access():
    """Ensure protected transaction route requires valid Bearer token."""
    response = client.get("/api/v1/transactions")
    assert response.status_code in (401, 403)
