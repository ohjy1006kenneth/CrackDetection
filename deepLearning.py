# U-Net
import torch
import torch.nn as nn
import torch.nn.functional as F

# Very basic U-Net block
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 1, 2, stride=2), nn.Sigmoid())

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.dec1(x2)
        return x3

# Training loop (simplified)
model = UNet().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Assume you have train_loader returning image, mask tensors
for epoch in range(5):
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
