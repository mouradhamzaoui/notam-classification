"""
Tests d'intégration — FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestHealthEndpoints:
    def test_health_check_200(self, test_client: TestClient):
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self, test_client: TestClient):
        data = test_client.get("/api/v1/health").json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime_s" in data

    def test_root_endpoint(self, test_client: TestClient):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "docs" in data

    def test_docs_available(self, test_client: TestClient):
        response = test_client.get("/docs")
        assert response.status_code == 200


@pytest.mark.integration
class TestClassifyEndpoint:
    def test_classify_valid_notam(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/classify",
            json={"text": "RWY 28L CLSD DUE TO MAINTENANCE"},
        )
        assert response.status_code == 200

    def test_classify_response_schema(self, test_client: TestClient):
        data = test_client.post(
            "/api/v1/classify",
            json={"text": "ILS RWY 28R NOT AVAILABLE"},
        ).json()
        assert "category" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "latency_ms" in data
        assert "priority" in data

    def test_classify_confidence_range(self, test_client: TestClient):
        data = test_client.post(
            "/api/v1/classify",
            json={"text": "PAPI RWY 36 OTS"},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_classify_probabilities_sum_to_one(self, test_client: TestClient):
        data = test_client.post(
            "/api/v1/classify",
            json={"text": "RWY 10L CLSD"},
        ).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_classify_category_is_valid(self, test_client: TestClient):
        valid_categories = {
            "RUNWAY_CLOSURE",
            "NAVIGATION_AID",
            "AIRSPACE_RESTRICTION",
            "LIGHTING",
            "OBSTACLE",
            "AERODROME_PROCEDURE",
        }
        data = test_client.post(
            "/api/v1/classify",
            json={"text": "RWY 28L CLSD"},
        ).json()
        assert data["category"] in valid_categories

    def test_classify_empty_text_rejected(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/classify",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_classify_text_too_short_rejected(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/classify",
            json={"text": "AB"},
        )
        assert response.status_code == 422

    def test_classify_text_uppercased(self, test_client: TestClient):
        """Vérifie que le texte est normalisé en majuscules."""
        r1 = test_client.post("/api/v1/classify", json={"text": "rwy 28l clsd"}).json()
        r2 = test_client.post("/api/v1/classify", json={"text": "RWY 28L CLSD"}).json()
        assert r1["category"] == r2["category"]


@pytest.mark.integration
class TestBatchClassifyEndpoint:
    def test_batch_valid(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/classify/batch",
            json={
                "texts": [
                    "RWY 10L CLSD DUE TO MAINTENANCE",
                    "ILS RWY 28R NOT AVAILABLE",
                    "PAPI RWY 18 OTS",
                ]
            },
        )
        assert response.status_code == 200

    def test_batch_response_count(self, test_client: TestClient):
        texts = ["RWY CLSD", "ILS OTS", "RESTRICTED AREA ACTIVE"]
        data = test_client.post(
            "/api/v1/classify/batch",
            json={"texts": texts},
        ).json()
        assert data["total"] == len(texts)
        assert len(data["results"]) == len(texts)

    def test_batch_duration_present(self, test_client: TestClient):
        data = test_client.post(
            "/api/v1/classify/batch",
            json={"texts": ["RWY CLSD", "ILS OTS"]},
        ).json()
        assert "duration_ms" in data
        assert data["duration_ms"] > 0

    def test_batch_empty_list_rejected(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/classify/batch",
            json={"texts": []},
        )
        assert response.status_code == 422

    def test_batch_over_limit_rejected(self, test_client: TestClient):
        texts = ["RWY CLSD"] * 101
        response = test_client.post(
            "/api/v1/classify/batch",
            json={"texts": texts},
        )
        assert response.status_code in [422, 500]


@pytest.mark.integration
class TestMonitoringEndpoints:
    def test_predictions_endpoint(self, test_client: TestClient):
        response = test_client.get("/api/v1/monitoring/predictions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_stats_endpoint(self, test_client: TestClient):
        response = test_client.get("/api/v1/monitoring/stats")
        assert response.status_code == 200
