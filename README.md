# 🛣️ Road Surface Classification API

This project provides a **FastAPI-based web API** for managing and serving machine learning models that classify **road surface types** from images (e.g., asphalt, concrete, unpaved). It supports model uploads, inference execution, performance monitoring, and job history tracking.

## 🚀 Features

- 🔄 **Model Management**: Upload, enable, disable, and list models
- 🧠 **Model Inference**: Classify single or multiple images
- 📊 **Performance Monitoring**: Track inference time and memory usage
- 📝 **Job History**: Log all past requests for auditing or debugging
- 🧩 **Dynamic Architecture Detection**: Automatically detect model type (SimpleCNN or EnhancedCNN)

---

## 📦 Directory Structure

```
.
├── app/                    # Main API module
│   ├── main.py             # FastAPI routes
│   └── processors/
│       └── model_processing.py   # ModelProcessor class
├── models/                 # Active models
├── models/archive/         # Archived models
├── testing/test_images/    # Sample images for testing
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📌 Endpoints

### 🔹 Model Management

| Method | Endpoint                             | Description                          |
|--------|--------------------------------------|--------------------------------------|
| POST   | `/models/upload`                     | Upload a new `.pth` model            |
| POST   | `/models/enable/{model_name}`        | Enable (activate) a model            |
| POST   | `/models/disable/{model_name}`       | Disable (archive) a model            |
| GET    | `/models`                            | List all active models               |
| GET    | `/models/archived`                   | List archived models                 |
| GET    | `/models/metadata/{model_name}`      | Get model architecture & metadata    |
| GET    | `/models/load/{model_name}`          | Load a model into memory             |

### 🔹 Inference

| Method | Endpoint                             | Description                          |
|--------|--------------------------------------|--------------------------------------|
| POST   | `/inference/execution`               | Run inference on a single image      |
| POST   | `/inference/batch_execution`         | Run inference on multiple images     |

### 🔹 Monitoring

| Method | Endpoint                             | Description                          |
|--------|--------------------------------------|--------------------------------------|
| GET    | `/performance`                       | Get last inference time & memory     |
| GET    | `/jobs/history?limit=n`              | Get job history (optional limit)     |

---

## 🧠 Supported Architectures

- **SimpleCNN**: A lightweight custom convolutional model
- **EnhancedCNN**: Based on ResNet50 with fine-tuned classification head

Models are auto-identified by architecture during upload.

---

## 🔧 Usage

### ✅ Requirements

- Python 3.8+
- pip

### 📦 Installation

```bash
git clone https://github.com/your-repo/road-classification-api.git
cd road-classification-api
pip install -r requirements.txt
```

### ▶️ Run the API

```bash
uvicorn app.main:app --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## 🧪 Testing the API

Use the provided Jupyter Notebook `api_demo.ipynb` for:

- Uploading models
- Running inference on test images
- Visualizing results with `matplotlib`

---

## 🛠️ Model Metadata Format (Auto-generated)

```json
{
  "model_name": "simple_model",
  "architecture": "SimpleCNN"
}
```

---

## Acknowledgments

Developed for the **Urban Digital Twin Interoperability Pilot (UDTIP)** led by the **Open Geospatial Consortium (OGC)**.
