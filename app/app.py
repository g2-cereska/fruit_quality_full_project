"""Simple Flask app for fruit quality classification."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from inference import Predictor


CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "outputs/best_model.pt")
UPLOAD_DIR = Path("app/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
predictor = None

HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fruit Quality Classifier</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            color: #1f2937;
        }
        .container {
            max-width: 1100px;
            margin: 32px auto;
            padding: 0 20px;
        }
        .card {
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 28px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }
        h1 {
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 2.5rem;
        }
        .muted {
            color: #6b7280;
            margin-bottom: 24px;
            font-size: 1.1rem;
        }
        .form-row {
            display: flex;
            gap: 14px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 18px;
        }
        input[type="file"] {
            padding: 10px;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            background: #fff;
        }
        button {
            padding: 12px 18px;
            border: none;
            border-radius: 10px;
            background: #111827;
            color: white;
            font-size: 1rem;
            cursor: pointer;
        }
        button:hover {
            background: #1f2937;
        }
        .divider {
            height: 1px;
            background: #e5e7eb;
            margin: 24px 0;
        }
        .result-grid {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 28px;
            align-items: start;
        }
        .image-box {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px;
            background: #fafafa;
        }
        .image-box img {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 10px;
        }
        .meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(120px, 1fr));
            gap: 12px;
            margin-bottom: 22px;
        }
        .meta-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 14px;
        }
        .meta-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 6px;
        }
        .meta-value {
            font-size: 1.15rem;
            font-weight: bold;
        }
        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: bold;
        }
        .fresh {
            background: #dcfce7;
            color: #166534;
        }
        .rotten {
            background: #fee2e2;
            color: #991b1b;
        }
        .prediction-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .prediction-list li {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 12px 14px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            margin-bottom: 10px;
            background: #fff;
        }
        .label-name {
            font-weight: 600;
        }
        .score {
            color: #111827;
            font-weight: bold;
            white-space: nowrap;
        }
        .error {
            margin-top: 16px;
            padding: 12px 14px;
            border-radius: 10px;
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        .footer-note {
            margin-top: 20px;
            color: #6b7280;
            font-size: 0.95rem;
        }
        @media (max-width: 850px) {
            .result-grid {
                grid-template-columns: 1fr;
            }
            .meta {
                grid-template-columns: 1fr;
            }
        }
        .score-high {
            color: #166534;
            font-weight: bold;
        }
        .score-medium {
            color: #92400e;
            font-weight: bold;
        }
        .score-low {
            color: #991b1b;
            font-weight: bold;
        }
        .bar-track {
            display: block;
            width: 100%;
            height: 12px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 6px;
        }
        .bar-fill {
            display: block;
            height: 100%;
            border-radius: 999px;
        }
        .bar-high {
            background: #22c55e;
        }
        .bar-medium {
            background: #f59e0b;
        }
        .bar-low {
            background: #ef4444;
        }
        .confidence-badge {
            display: inline-block;
            margin-top: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: bold;
        }
        .conf-high {
            background: #dcfce7;
            color: #166534;
        }
        .conf-medium {
            background: #fef3c7;
            color: #92400e;
        }
        .conf-low {
            background: #fee2e2;
            color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Fruit and Vegetable Quality Classifier</h1>
            <p class="muted">
                Upload an image to classify produce freshness, for example Apple Fresh or Tomato Rotten.
            </p>

            <form method="post" enctype="multipart/form-data">
                <div class="form-row">
                    <input type="file" name="image" accept="image/*" required>
                    <button type="submit">Predict</button>
                </div>
            </form>

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            {% if result %}
                <div class="divider"></div>
                <h2>Prediction Result</h2>

                <div class="result-grid">
                    <div class="image-box">
                        {% if image_path %}
                            <img src="{{ image_path }}" alt="Uploaded image">
                        {% endif %}
                    </div>

                    <div>
                        <div class="meta">
                            <div class="meta-card">
                                <div class="meta-label">Best class</div>
                                <div class="meta-value">{{ result.best_label|replace("_", " ") }}</div>

                                {% set best_prob = result.top_predictions[0].probability %}
                                <span class="confidence-badge
                                    {% if best_prob >= 0.8 %}
                                        conf-high
                                    {% elif best_prob >= 0.5 %}
                                        conf-medium
                                    {% else %}
                                        conf-low
                                    {% endif %}
                                ">
                                    {% if best_prob >= 0.8 %}
                                        High confidence
                                    {% elif best_prob >= 0.5 %}
                                        Medium confidence
                                    {% else %}
                                        Low confidence
                                    {% endif %}
                                </span>
                            </div>
                            <div class="meta-card">
                                <div class="meta-label">Produce</div>
                                <div class="meta-value">{{ result.produce_name|replace("_", " ") }}</div>
                            </div>
                            <div class="meta-card">
                                <div class="meta-label">Condition</div>
                                <div class="meta-value">
                                    <span class="pill {% if result.condition|lower == 'fresh' %}fresh{% else %}rotten{% endif %}">
                                        {{ result.condition }}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <h3>Top predictions</h3>
                        <ul class="prediction-list">
                        {% for item in result.top_predictions %}
                            <li style="display:block;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span class="label-name">{{ item.label|replace("_", " ") }}</span>

                                    <span class="score
                                        {% if item.probability >= 0.8 %}
                                            score-high
                                        {% elif item.probability >= 0.5 %}
                                            score-medium
                                        {% else %}
                                            score-low
                                        {% endif %}
                                    ">
                                        {{ '%.2f'|format(item.probability * 100) }}%
                                    </span>
                                </div>

                                <div class="bar-track">
                                    <div
                                        class="bar-fill
                                            {% if item.probability >= 0.8 %}
                                                bar-high
                                            {% elif item.probability >= 0.5 %}
                                                bar-medium
                                            {% else %}
                                                bar-low
                                            {% endif %}
                                        "
                                        style="width: {{ '%.2f'|format(item.probability * 100) }}%;"
                                    ></div>
                                </div>
                            </li>
                        {% endfor %}
                        </ul>

                        <p class="footer-note">
                            This result is a model prediction and may be less reliable for images outside the training distribution.
                        </p>
                    </div>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


def get_predictor() -> Predictor:
    global predictor
    if predictor is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at {CHECKPOINT_PATH}. Train the model first or set MODEL_PATH."
            )
        predictor = Predictor(CHECKPOINT_PATH)
    return predictor


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    image_path = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or uploaded.filename == "":
            error = "Please select an image file."
        else:
            safe_name = secure_filename(uploaded.filename)
            if not safe_name:
                safe_name = "uploaded_image.png"

            save_path = UPLOAD_DIR / safe_name
            uploaded.save(save_path)

            try:
                result = get_predictor().predict(save_path)
                image_path = f"/static/uploads/{safe_name}"
            except Exception as exc:  # pragma: no cover - app path
                error = str(exc)

    return render_template_string(
        HTML,
        result=result,
        error=error,
        image_path=image_path,
    )


@app.route("/health")
def health():
    status = {"status": "ok", "model_loaded": os.path.exists(CHECKPOINT_PATH)}
    return jsonify(status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)