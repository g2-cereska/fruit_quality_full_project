from io import BytesIO

import app.app as app_module


class DummyPredictor:
    def predict(self, image_path):
        return {
            "best_label": "Apple__Fresh",
            "produce_name": "Apple",
            "condition": "Fresh",
            "top_predictions": [
                {"label": "Apple__Fresh", "probability": 0.91},
                {"label": "Banana__Fresh", "probability": 0.06},
                {"label": "Tomato__Rotten", "probability": 0.03},
            ],
        }


def test_health_endpoint():
    client = app_module.app.test_client()
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.get_json()
    assert "status" in payload
    assert "model_loaded" in payload
    assert payload["status"] == "ok"


def test_home_get_returns_page():
    client = app_module.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Fruit and Vegetable Quality Classifier" in response.data


def test_home_post_without_file_shows_error():
    client = app_module.app.test_client()
    response = client.post("/", data={}, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b"Please select an image file." in response.data


def test_home_post_with_mock_predictor(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(app_module, "get_predictor", lambda: DummyPredictor())

    client = app_module.app.test_client()

    data = {
        "image": (BytesIO(b"fake image bytes"), "test.jpg"),
    }

    response = client.post("/", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b"Prediction Result" in response.data
    assert b"Apple" in response.data
    assert b"Fresh" in response.data
    assert b"High confidence" in response.data