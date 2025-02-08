import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
import psutil
from pynvml import *
from sklearn.preprocessing import LabelEncoder

# Check if CUDA is available for GPU acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def print_gpu_info():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    utilization = nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Utilization: {utilization.gpu}%, Memory Usage: {info.used//1024**2} MB / {info.total//1024**2} MB")
    nvmlShutdown()

# Custom Dataset class to load images from CSV with labels for conditional generation
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # Use LabelEncoder to convert string labels to integer labels
        self.label_encoder = LabelEncoder()
        self.data['style_encoded'] = self.label_encoder.fit_transform(self.data['style'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx]['style_encoded']
        return image, label

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
dataset = ImageDataset('dataset.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Generator Model for conditional GAN
class Generator(nn.Module):
    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 100)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(200, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        labels = labels.long()
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((z, label_embedding.unsqueeze(2).unsqueeze(3).expand(z.size(0), 100, z.size(2), z.size(3))), 1)
        return self.gen(gen_input)

# Discriminator Model for conditional GAN
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 3 * 128 * 128)
        self.dis = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 8, 1, 0, bias=False),
        )

    def forward(self, img, labels):
        labels = labels.long()
        label_embedding = self.label_emb(labels).view(img.size(0), 3, 128, 128)
        dis_input = torch.cat((img, label_embedding), 1)
        output = self.dis(dis_input)
        return torch.sigmoid(output.view(-1, 1))

# Determine number of unique styles from the dataset
num_classes = len(dataset.label_encoder.classes_)

# Initialize models
generator = Generator(num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)
print(f"Generator is on: {next(generator.parameters()).device}")
print(f"Discriminator is on: {next(discriminator.parameters()).device}")

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Add learning rate schedulers
schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.1)
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(dataloader):
        real_data = data.to(device)
        labels = labels.to(device).long()
        batch_size = real_data.size(0)

        # Debug print statements
        if i % 10 == 0:  # Print every 10th batch
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(dataloader)}, CPU: {psutil.cpu_percent()}%")
            print_gpu_info()
        
        # Train Discriminator with real images
        discriminator.zero_grad()
        label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float)
        output = discriminator(real_data, labels)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train Discriminator with fake images
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = generator(noise, labels)
        label.fill_(0)
        output = discriminator(fake.detach(), labels)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        schedulerD.step()

        # Train Generator
        generator.zero_grad()
        label.fill_(1)  # Fake labels are real for generator cost
        output = discriminator(fake, labels)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        schedulerG.step()

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z2:.4f}')

# Generate a sample image after training with a specific style
with torch.no_grad():
    noise = torch.randn(1, 100, 1, 1, device=device)
    style_index = 0  # Example: 0 might correspond to 'dark'
    labels = torch.tensor([style_index], device=device).long()
    fake = generator(noise, labels).detach().cpu()
    img = transforms.ToPILImage()(fake[0] * 0.5 + 0.5)
    img.save('generated_image_with_specific_style.png')

print("Generated image saved as 'generated_image_with_specific_style.png'")