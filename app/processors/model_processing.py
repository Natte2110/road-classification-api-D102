import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import io
import time
import json
import psutil
import shutil
from fastapi import UploadFile

# Define directories
MODELS_DIR = "./models"
ARCHIVED_MODELS_DIR = "./models/archive"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARCHIVED_MODELS_DIR, exist_ok=True)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Class mapping for road surfaces
class_mapping = {0: 'asphalt', 1: 'concrete', 2: 'unpaved'}

# Define the SimpleCNN Model Structure
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Define an Enhanced Model using ResNet-50
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(EnhancedCNN, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Define the Model Processor Class
class ModelProcessor:
    """
    Handles model loading, uploading, architecture detection, enabling/disabling models,
    image preprocessing, and inference for road surface classification.
    """

    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.last_inference_time = None
        self.last_memory_usage = None
        self.model_accuracy = None

    def upload_model(self, uploaded_file: UploadFile):
        """
        Saves a model file and detects its architecture.
        """
        model_path = os.path.join(ARCHIVED_MODELS_DIR, uploaded_file.filename)

        # Save model file
        with open(model_path, "wb") as f:
            f.write(uploaded_file.file.read())

        # Detect architecture
        detected_architecture = self.detect_architecture(model_path)

        # Save model metadata
        model_metadata = {
            "model_name": uploaded_file.filename.replace(".pth", ""),
            "architecture": detected_architecture
        }
        metadata_path = model_path.replace(".pth", ".json")
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f)

        return {
            "message": f"Model '{uploaded_file.filename}' uploaded successfully.",
            "architecture_detected": detected_architecture
        }

    def detect_architecture(self, model_path):
        """
        Detects the architecture of the uploaded model.
        """
        try:
            model_state = torch.load(model_path, map_location=torch.device('cpu'))

            if isinstance(model_state, dict) and "state_dict" in model_state:
                model_state = model_state["state_dict"]

            if any(key.startswith("fc") or "layer4" in key for key in model_state.keys()):
                return "EnhancedCNN"
            elif any(key.startswith("classifier") for key in model_state.keys()):
                return "SimpleCNN"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"

    def enable_model(self, model_name):
        """
        Moves a model and its metadata from archive to active models directory.
        """
        archive_model_path = os.path.join(ARCHIVED_MODELS_DIR, f"{model_name}.pth")
        active_model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")

        archive_metadata_path = archive_model_path.replace(".pth", ".json")
        active_metadata_path = active_model_path.replace(".pth", ".json")

        if not os.path.exists(archive_model_path):
            raise FileNotFoundError(f"Model '{model_name}' not found in archive.")

        # Move model and metadata if it exists
        shutil.move(archive_model_path, active_model_path)
        
        if os.path.exists(archive_metadata_path):  # Ensure metadata exists before moving
            shutil.move(archive_metadata_path, active_metadata_path)

        return {"message": f"Model '{model_name}' is now active."}


    def disable_model(self, model_name):
        """
        Moves a model and its metadata from active models directory to archive.
        """
        active_model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        archive_model_path = os.path.join(ARCHIVED_MODELS_DIR, f"{model_name}.pth")

        active_metadata_path = active_model_path.replace(".pth", ".json")
        archive_metadata_path = archive_model_path.replace(".pth", ".json")

        if not os.path.exists(active_model_path):
            raise FileNotFoundError(f"Model '{model_name}' not found in active models.")

        # Move model and metadata if it exists
        shutil.move(active_model_path, archive_model_path)
        
        if os.path.exists(active_metadata_path):  # Ensure metadata exists before moving
            shutil.move(active_metadata_path, archive_metadata_path)

        return {"message": f"Model '{model_name}' has been disabled and moved to archive."}


    def list_models(self, archived=False):
        """
        Lists models in either the active or archive directory.
        """
        directory = ARCHIVED_MODELS_DIR if archived else MODELS_DIR
        return [file.replace(".pth", "") for file in os.listdir(directory) if file.endswith(".pth")]

    def get_model_details(self, model_name):
        """
        Returns metadata about a model, including:
        - Architecture
        - Number of parameters
        - File size
        """
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_name}' not found.")

        architecture = self.detect_architecture(model_path)
        model = SimpleCNN() if architecture == "SimpleCNN" else EnhancedCNN()
        num_params = sum(p.numel() for p in model.parameters())
        file_size = os.path.getsize(model_path) / (1024 * 1024)

        return {
            "model_name": model_name,
            "architecture": architecture,
            "num_parameters": num_params,
            "file_size_mb": round(file_size, 2)
        }


    def load_model(self, model_name: str):
        """
        Dynamically loads a model using metadata to determine the correct architecture.
        """
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        metadata_path = model_path.replace(".pth", ".json")  # Metadata file

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model '{model_name}' or its metadata not found.")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        architecture = metadata.get("architecture", "")

        # Dynamically select the correct model class
        if architecture.lower() == "simplecnn":
            self.model = SimpleCNN()
        elif architecture.lower() == "enhancedcnn":
            self.model = EnhancedCNN()
        else:
            raise ValueError(f"Unknown architecture '{architecture}' for model '{model_name}'.")

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.current_model_name = model_name


    def preprocess_image(self, image_data: bytes):
        """
        Reads and preprocesses an image from bytes.
        """
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        return transform(img).unsqueeze(0)  # Add batch dimension

    def run_inference(self, image_data: bytes):
        """
        Performs inference on an image and logs performance metrics.
        """
        if self.model is None:
            raise RuntimeError("No model is loaded. Please call 'load_model()' first.")

        # Start timing and measure memory usage before inference
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Convert to MB

        # Preprocess the image
        input_tensor = self.preprocess_image(image_data)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor).squeeze(0).softmax(0)
            class_id = outputs.argmax().item()
            confidence = outputs[class_id].item()

        # Stop timing and measure memory after inference
        end_time = time.time()
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Convert to MB

        # Update performance metrics
        self.last_inference_time = round((end_time - start_time) * 1000, 2)  # Convert to ms
        self.last_memory_usage = round(memory_after - memory_before, 2)  # MB difference

        return {
            "prediction": class_mapping.get(class_id, "unknown"),
            "confidence": f"{100 * confidence:.1f}%"
        }

    def get_performance_metrics(self):
        """
        Returns the last recorded performance metrics.
        """
        return {
            "inference_time": f"{self.last_inference_time} ms" if self.last_inference_time else "N/A",
            "memory_usage": f"{self.last_memory_usage} MB" if self.last_memory_usage else "N/A",
            "model_accuracy": f"{self.model_accuracy}%" if self.model_accuracy else "N/A"
        }
