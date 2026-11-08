import pytest
from fastapi.testclient import TestClient

from web.main import SERVICE_REGISTRY, app


@pytest.fixture(autouse=True)
def clear_service_registry():
    SERVICE_REGISTRY.shutdown()
    yield
    SERVICE_REGISTRY.shutdown()


def test_browser_sessions_do_not_share_processing_configuration():
    with TestClient(app) as first_client, TestClient(app) as second_client:
        assert first_client.get("/").status_code == 200
        assert second_client.get("/").status_code == 200

        first_session = first_client.cookies.get("rppg_session")
        second_session = second_client.cookies.get("rppg_session")
        assert first_session
        assert second_session
        assert first_session != second_session

        updated = first_client.post("/api/config", json={"segLength": 512})
        assert updated.status_code == 200

        assert first_client.get("/api/config").json()["segLength"] == 512
        assert second_client.get("/api/config").json()["segLength"] == 1024


def test_healthcheck_does_not_allocate_a_processing_service():
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert SERVICE_REGISTRY.service_count == 0