
from fastapi.testclient import TestClient

from app.main import app, require_api_key


# Bypass API-key authentication only within this test client.
# Production authentication remains unchanged.
app.dependency_overrides[require_api_key] = lambda: None

client = TestClient(app)


def test_valid_classification_returns_200() -> None:
    payload = {
        "category": "API Integration",
        "priority": "High",
        "summary": "The customer needs an external API connected.",
        "next_action": "Collect the API documentation and requirements.",
    }

    response = client.post(
        "/validate-classification",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json() == payload


def test_invalid_priority_returns_422() -> None:
    payload = {
        "category": "API Integration",
        "priority": "Urgent",
        "summary": "The customer needs an API integration.",
        "next_action": "Review the integration requirements.",
    }

    response = client.post(
        "/validate-classification",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_category_returns_422() -> None:
    payload = {
        "category": "Graphic Design",
        "priority": "Medium",
        "summary": "The customer has a service request.",
        "next_action": "Clarify the requested service.",
    }

    response = client.post(
        "/validate-classification",
        json=payload,
    )

    assert response.status_code == 422


def test_empty_summary_returns_422() -> None:
    payload = {
        "category": "Technical Support",
        "priority": "Low",
        "summary": "",
        "next_action": "Ask the customer for more information.",
    }

    response = client.post(
        "/validate-classification",
        json=payload,
    )

    assert response.status_code == 422


if __name__ == "__main__":
    try:
        test_valid_classification_returns_200()
        test_invalid_priority_returns_422()
        test_invalid_category_returns_422()
        test_empty_summary_returns_422()

        print("All AI classification validation tests passed.")

    finally:
        app.dependency_overrides.clear()
        client.close()
