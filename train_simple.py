import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            for image_name in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, image_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label


# Define transformation (same as the encoder preprocessing)
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CustomDataset(root_dir="/path/to/dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


class ClassificationHead(nn.Module):
    def __init__(self, input_channels=256, num_classes=256, hidden_dim=768):
        super(ClassificationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(512, hidden_dim)  # Hidden fully connected layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # Output layer for classification

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten to (BATCH_SIZE, 512)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


encoder = predictor.model.image_encoder
head = ClassificationHead(input_channels=256, num_classes=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
head = head.to(device)
encoder.eval()  # Freeze the encoder
for param in encoder.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(head.parameters(), lr=0.001)

writer = SummaryWriter()  # TensorBoard writer
best_loss = float('inf')

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = encoder(images)  # Extract features from frozen encoder

        outputs = head(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save the model if loss improves
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(head.state_dict(), save_path)
        print(f"Model saved with loss {best_loss:.4f}")

writer.close()
