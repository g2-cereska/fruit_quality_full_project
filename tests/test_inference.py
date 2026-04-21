from pathlib import Path

import torch
from PIL import Image

import inference


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # 3 classes, class index 0 should win
        batch_size = x.shape[0]
        return torch.tensor([[3.0, 1.0, 0.5]] * batch_size)


def fake_load_checkpoint(checkpoint_path, device):
    model = DummyModel().to(device)
    class_names = ["Apple__Fresh", "Banana__Rotten", "Carrot__Fresh"]
    history = {"accuracy": [0.9]}
    return model, class_names, history


def create_temp_image(path: Path) -> None:
    image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    image.save(path)


def test_predictor_initialises(monkeypatch):
    monkeypatch.setattr(inference, "load_checkpoint", fake_load_checkpoint)

    predictor = inference.Predictor("fake_checkpoint.pt")

    assert predictor is not None
    assert predictor.class_names == ["Apple__Fresh", "Banana__Rotten", "Carrot__Fresh"]
    assert predictor.device.type in {"cpu", "cuda"}


def test_predict_returns_expected_structure(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "load_checkpoint", fake_load_checkpoint)

    image_path = tmp_path / "sample.jpg"
    create_temp_image(image_path)

    predictor = inference.Predictor("fake_checkpoint.pt")
    result = predictor.predict(image_path)

    assert isinstance(result, dict)
    assert "best_label" in result
    assert "produce_name" in result
    assert "condition" in result
    assert "top_predictions" in result

    assert result["best_label"] == "Apple__Fresh"
    assert result["produce_name"] == "Apple"
    assert result["condition"] == "Fresh"

    top_predictions = result["top_predictions"]
    assert isinstance(top_predictions, list)
    assert len(top_predictions) == 3

    for item in top_predictions:
        assert "label" in item
        assert "probability" in item
        assert isinstance(item["label"], str)
        assert 0.0 <= item["probability"] <= 1.0


def test_predict_invalid_image_path_raises(monkeypatch):
    monkeypatch.setattr(inference, "load_checkpoint", fake_load_checkpoint)

    predictor = inference.Predictor("fake_checkpoint.pt")

    missing_path = "this_file_does_not_exist.jpg"

    try:
        predictor.predict(missing_path)
        assert False, "Expected an exception for a missing image path"
    except FileNotFoundError:
        assert True