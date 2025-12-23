# Shipcube AI Email Processing

AI-powered email processing system for Shipcube logistics platform.

## 📁 Project Structure

```
shipcube-AIemail/
├── .venv/                  # Virtual environment
├── data/                   # Training and test data
│   ├── shipcube_train.csv  # Training dataset
│   ├── shipcube_val.csv    # Validation dataset
│   └── shipcube_test.csv   # Test dataset
├── src/                    # Source code
│   ├── __init__.py
│   ├── prototype_infer.py  # Prototype inference script
│   ├── train_lora.py       # LoRA training script
│   └── evaluate.py         # Model evaluation script
├── server.py               # Flask API server
├── ui.py                   # Streamlit UI application
└── README.md
```

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install torch transformers datasets accelerate peft bitsandbytes
pip install flask flask-cors streamlit requests
pip install pandas numpy scikit-learn rouge-score
```

### 3. Setup HuggingFace Authentication

The model requires HuggingFace authentication:

```bash
huggingface-cli login
```

Or in Python:
```python
from huggingface_hub import login
login()
```

Get your token from: https://huggingface.co/settings/tokens

### 4. Run Prototype Inference

```bash
python src/prototype_infer.py
```

### 5. Train Model

```bash
python src/train_lora.py \
    --train_file data/shipcube_train.csv \
    --val_file data/shipcube_val.csv \
    --output_dir checkpoints/shipcube-lora \
    --epochs 3 \
    --batch_size 4
```

### 6. Evaluate Model

```bash
python src/evaluate.py \
    --model_path checkpoints/shipcube-lora \
    --test_file data/shipcube_test.csv \
    --output evaluation_results.json
```

### 7. Run API Server

```bash
python server.py
```

The server will start on `http://localhost:5000`

### 8. Run UI Application

```bash
streamlit run ui.py
```

The UI will open in your browser at `http://localhost:8501`

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Single Email Processing
```bash
POST /predict
Content-Type: application/json

{
    "email": "Hi team, my shipment SCB1234 hasn't arrived. Please update ETA."
}
```

### Batch Email Processing
```bash
POST /batch
Content-Type: application/json

{
    "emails": ["email1", "email2", ...]
}
```

## 📝 Data Format

CSV files should have the following columns:
- `email`: Email content
- `query`: Extracted query
- `intent`: Intent classification (track_shipment, track_order, complaint, etc.)
- `sentiment`: Sentiment (positive, negative, neutral)
- `priority`: Priority level (high, medium, low)
- `entities`: JSON string of extracted entities
- `response`: Automated response

## 🧠 Model Details

- **Base Model**: `iprajwaal/gemma-3b-chat-support`
- **Training Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.1

## 📈 Evaluation Metrics

The evaluation script calculates:
- Intent Accuracy
- Intent F1 Score
- Sentiment Accuracy
- ROUGE-1 Score
- ROUGE-L Score

## 🔧 Configuration

Training parameters can be adjusted in `src/train_lora.py`:
- Learning rate
- Batch size
- Number of epochs
- LoRA configuration

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

