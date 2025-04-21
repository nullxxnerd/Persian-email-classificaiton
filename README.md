# Yekta Project - TextCNN API

This project provides a FastAPI service for classifying Persian text using a trained TextCNN model.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn torch tensorflow keras
   ```

3. Ensure model artifacts are in the project root:
   - `textcnn.pth` (model state)
   - `tokenizer.pkl` (tokenizer object)
   - `config.pkl` (config with `max_len`, `label_map_inv`, `stop_words`)

If you haven't exported them yet, run:
```bash
python export_model.py
```

## Running the API

Start the server with Uvicorn:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## `/predict` Endpoint

### Request

- **Method**: `POST`
- **URL**: `/predict`
- **Headers**:
  - `Content-Type: application/json`
- **Body**:
  ```json
  {
    "text": "متنی که می‌خواهید طبقه‌بندی شود"
  }
  ```

### Response

- **Status**: `200 OK`
- **Body**:
  ```json
  {
    "label": "sales_inquiry",
    "probabilities": {
      "customer_support": 0.05,
      "sales_inquiry": 0.90,
      "partnership": 0.03,
      "spam": 0.02
    }
  }
  ```

- `label`: Predicted class label
- `probabilities`: Softmax scores for each class

## Logging

Training logs are written to `training.log` during model training.

---

Feel free to open issues or request additional features!
