import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LineDataset
from unet import UNet
import torchvision.transforms as transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load dataset
image_dir = 'dataset/images'
mask_dir = 'dataset/masks'
dataset = LineDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the model, loss function, and optimizer
model = UNet(in_channels=3, out_channels=1).cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint_path = 'unet_line_detection(2).pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Checkpoint loaded. Continuing training...")

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for images, masks in dataloader:
        images, masks = images.cuda(), masks.cuda()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'unet_line_detection(2).pth')
