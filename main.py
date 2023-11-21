import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
# Set random seeds for reproducibility
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True



def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total

    return loss, accuracy

class DeeperNet(nn.Module):
    def __init__(self):
        super(DeeperNet, self).__init__()
        # Define the first convolutional layer with 3 input channels, 32 output channels, a 3x3 kernel, and 1 pixel padding.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Batch normalization applied to the output of the first convolutional layer with 32 channels.
        self.bn1 = nn.BatchNorm2d(32)

        # Define the second convolutional layer with 32 input channels, 64 output channels, a 3x3 kernel, and 1 pixel padding.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Batch normalization applied to the output of the second convolutional layer with 64 channels.
        self.bn2 = nn.BatchNorm2d(64)

        # Define the third convolutional layer with 64 input channels, 128 output channels, a 3x3 kernel, and 1 pixel padding.
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch normalization applied to the output of the third convolutional layer with 128 channels.
        self.bn3 = nn.BatchNorm2d(128)

        # Define the first fully connected (linear) layer with input size of 128 * 4 * 4 and output size of 512.
        self.fc1 = nn.Linear(128 * 4 * 4, 512)

        # Define the second fully connected (linear) layer with input size of 512 and output size of 10 (number of classes).
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_embeddings(self, x):
        # Extract embeddings from the second-to-last fully connected layer (fc1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        # The embeddings are the output of the fc1 layer
        embeddings = x

        return embeddings

def create_data_loaders(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DeeperNet()
    net.to(device)

    batch_size = 64
    num_workers = 2 if torch.cuda.is_available() else 0

    train_loader, test_loader = create_data_loaders(batch_size, num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    # Lists to store losses and accuracies
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Lists to store embeddings
    train_embeddings = []

    # Training loop
    num_epochs = 20

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Extract embeddings and save them
            embeddings = net.get_embeddings(inputs)
            train_embeddings.extend(embeddings.cpu().detach().numpy())

        # Update the learning rate
        scheduler.step(running_loss)

        # Evaluate on training and test set
        train_loss, train_accuracy = evaluate_model(net, train_loader, criterion, device)
        test_loss, test_accuracy = evaluate_model(net, test_loader, criterion, device)

        # Store losses and accuracies
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print statistics every epoch
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    train_embeddings = np.array(train_embeddings)
    np.save('train_embeddings_'+ current_date + '.npy', train_embeddings)

    # Plotting Cross-Entropy Loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='orange')
    plt.title('Cross-Entropy Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Classification Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='green')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy', color='red')
    plt.title('Classification Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_file_name = f'chart_{current_date}.png'
    plt.savefig(plot_file_name)
    plt.show()

main()