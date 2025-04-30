import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # Progress bar
import torchvision.models as models

# Step 1: Custom Dataset with High-Level Class Mapping
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # High-level class mapping
        self.class_map = {
            'asphalt': ['asphalt'],
            'concrete': ['concrete'],
            'unpaved': ['gravel', 'mud', 'snow', 'ice']
        }
        self.class_to_idx = {'asphalt': 0, 'concrete': 1, 'unpaved': 2}
        
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('jpg', 'png', 'jpeg')):
                    label = self.get_high_level_label(file)
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(self.class_to_idx[label])
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
                    
    def get_high_level_label(self, file_name):
        for high_label, keywords in self.class_map.items():
            for keyword in keywords:
                if keyword in file_name.lower():
                    return high_label
        return 'unpaved'  # Default to unpaved
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 2: Data Augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

vali_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Step 3: Load Datasets
train_dir = '/mnt/d/RSCD dataset-1million/RSCD dataset-1million/train'
vali_dir = '/mnt/d/RSCD dataset-1million/RSCD dataset-1million/vali_20k'

train_dataset = CustomImageDataset(root_dir=train_dir, transform=train_transforms)
vali_dataset = CustomImageDataset(root_dir=vali_dir, transform=vali_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
vali_loader = DataLoader(vali_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

# Step 4: Define Model (ResNet-50 with Modified Classifier)
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(EnhancedCNN, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased dropout to reduce overfitting
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN(num_classes=3).to(device)
assert next(model.parameters()).is_cuda, "Model is not on GPU!"

# Step 5: Define Loss & Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Adjusted label smoothing
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()

# Step 6: Training Loop with Mixed Precision and Performance Optimizations
early_stop_count = 0
best_accuracy = 0.0
num_epochs = 30  # Extended training epochs for fine-tuning
patience = 5

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")
    for inputs, labels in train_progress:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        train_progress.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")
    scheduler.step()
    
    if avg_loss < best_accuracy:
        best_accuracy = avg_loss
        early_stop_count = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        print("Model improved and saved.")
    else:
        early_stop_count += 1
    
    if early_stop_count >= patience:
        print("Early stopping triggered.")
        torch.save(model.state_dict(), 'best_model.pth')
        break
