# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress bar

# Step 1: Custom Dataset for High-Level Class Mapping
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
        
        # Recursively load images from subfolders
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('jpg', 'png', 'jpeg')):  # Supported image formats
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

# Step 2: Define Transforms for Training and Validation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Step 3: Setup Directories and Load Datasets
train_dir = '/mnt/d/03_SoftwareDevelopment/00_Data/RSCD dataset-1million/train'
vali_dir = '/mnt/d/03_SoftwareDevelopment/00_Data/RSCD dataset-1million/vali_20k'

print("Initializing training and validation datasets...")
train_dataset = CustomImageDataset(root_dir=train_dir, transform=data_transforms)
vali_dataset = CustomImageDataset(root_dir=vali_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
vali_loader = DataLoader(vali_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(vali_dataset)}")

# Step 4: Define the Model - Simple CNN Classifier
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
            nn.Linear(64 * 28 * 28, 128),  # Assuming image size 224x224
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # 3 classes: asphalt, concrete, unpaved
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Step 5: Initialize Model, Loss, and Optimizer
print("Initializing model, loss, and optimizer...")
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training and Validation Loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    
    # Training Loop with Progress Bar
    train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")
    for inputs, labels in train_progress:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_progress.set_postfix(loss=loss.item())
    
    print(f"Epoch {epoch+1} Training Loss: {running_loss/len(train_loader):.4f}")

    # Step 7: Validation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        vali_progress = tqdm(vali_loader, desc=f"Validating Epoch {epoch+1}", unit="batch")
        for inputs, labels in vali_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%")

print("Training Complete")

# Step 8: Save the Model
torch.save(model.state_dict(), 'classifier_model.pth')
print("Model saved as 'classifier_model.pth'")
