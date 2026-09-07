"""
Phase 0 smoke tests. Just proves the test harness works and the app boots.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_sanity():
    """Trivial test to confirm pytest itself is working."""
    assert 1 + 1 == 2


def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_expected_shape():
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "environment" in body
