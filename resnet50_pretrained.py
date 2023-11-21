import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models

# Set random seeds for reproducibility
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Define data transformations and create data loaders for CIFAR-10
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Assuming you have downloaded the CIFAR-10 dataset using torchvision.datasets.CIFAR10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Initialize the ResNet50 model
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.resnet50(x)

net = ResNet50()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduling
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
num_epochs = 50  # You can adjust this based on your needs

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

def evaluate_model(model, data_loader, losses_list, accuracies_list):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        losses_list.append(running_loss / len(data_loader))
        accuracies_list.append(100. * correct / total)


for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Update the learning rate
    scheduler.step()

    # Print statistics every epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Evaluate on training set
    net.eval()
    evaluate_model(net, train_loader, train_losses, train_accuracies)

    # Evaluate on test set
    evaluate_model(net, test_loader, test_losses, test_accuracies)

# Plotting Cross-Entropy Loss and Accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='orange')
plt.title('Cross-Entropy Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='green')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy', color='red')
plt.title('Classification Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
