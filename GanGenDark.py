import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import psutil
from pynvml import *

# Check if CUDA is available for GPU acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Define a simple model to test GPU usage
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.conv2(x)
        return x

# Dataset class for loading images from a single folder
class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.img_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize the model
model = SimpleModel().to(device)
print(f"Model is on: {next(model.parameters()).device}")

# Load dataset
dataset = SingleFolderDataset('img/dark', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Function to print GPU info
def print_gpu_info():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    utilization = nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Utilization: {utilization.gpu}%, Memory Usage: {info.used//1024**2} MB / {info.total//1024**2} MB")
    nvmlShutdown()

# Simple training loop to test GPU usage
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        inputs = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Using MSE loss to compare input and output
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if i % 1 == 0:  # Print every batch
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(dataloader)}, CPU: {psutil.cpu_percent()}%")
            print_gpu_info()

print("Finished testing GPU utilization with simple model on 'dark' images.")