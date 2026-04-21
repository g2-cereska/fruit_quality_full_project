from pathlib import Path


def test_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    expected = [
        root / "requirements.txt",
        root / "Dockerfile",
        root / "src" / "train.py",
        root / "src" / "data_utils.py",
        root / "src" / "model_utils.py",
        root / "src" / "inference.py",
        root / "app" / "app.py",
    ]
    for file_path in expected:
        assert file_path.exists(), f"Missing required file: {file_path}"
