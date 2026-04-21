# Fruit and Vegetable Quality Classification Service

This project implements a 28-class computer vision system for the Bristol Regional Food Network use case. It classifies a produce image into one of 28 labels such as `Apple__Healthy` or `Tomato__Rotten`, then exposes the result through a small Flask service that can be integrated into a wider Django or microservice-based marketplace.

## Why this fits the assignment
- Producers need automated quality assessment for fruits and vegetables.
- The AI solution must be deployable as a service with appropriate interfaces.
- The case study also expects explainability, monitoring, and support for model retraining by AI engineers.

## Project structure
- `src/train.py` – training pipeline with EfficientNet-B0 transfer learning
- `src/inference.py` – single-image prediction helper
- `src/data_utils.py` – dataset validation, transforms, and train/validation/test split
- `app/app.py` – Flask web app for uploading an image and getting a prediction
- `Dockerfile` – container packaging
- `assets/` – architecture diagram and benchmark figures used in the technical report
- `docs/technical_report.docx` – submission-ready technical report

## Expected dataset layout
Download the Kaggle dataset and point the training script to the folder that directly contains the 28 class folders.

```text
Fruit And Vegetable Diseases Dataset/
├── Apple__Healthy/
├── Apple__Rotten/
├── Banana__Healthy/
├── Banana__Rotten/
├── ...
└── Tomato__Rotten/
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model
```bash
python src/train.py --data_dir "C:\path\to\Fruit And Vegetable Diseases Dataset" --output_dir outputs --epochs 6 --batch_size 32
```

Artifacts produced after training:
- `outputs/best_model.pt`
- `outputs/loss_curve.png`
- `outputs/accuracy_curve.png`
- `outputs/confusion_matrix.png`
- `outputs/classification_report.csv`
- `outputs/metrics.json`

## Run the web app
```bash
set MODEL_PATH=outputs/best_model.pt
python app/app.py
```
Then open `http://127.0.0.1:5000`.

## Docker run
```bash
docker build -t fruit-quality-service .
docker run -p 5000:5000 -e MODEL_PATH=/app/outputs/best_model.pt fruit-quality-service
```

## Common training errors and fixes
### 1. Dataset folder not found
Use the folder that contains the 28 class folders, not the parent Kaggle download directory.

### 2. Broken or unreadable images
`SafeImageFolder` raises a friendly path-specific error. Remove or replace the bad file.

### 3. CUDA not available
The code automatically falls back to CPU.

### 4. Windows path with spaces
Wrap the dataset path in quotes.

### 5. Class mismatch after loading a checkpoint
Delete the old checkpoint and retrain if the class list changed.

## Recommended extensions
- Add Grad-CAM explanations for producer and administrator dashboards.
- Log user corrections when a producer overrides the prediction.
- Store inference events in PostgreSQL for drift monitoring and retraining.
- Add a discount recommendation rule layer based on predicted condition and product age.
