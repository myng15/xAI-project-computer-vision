if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt


    # Set random seeds for reproducibility
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 3 input image channels (for CIFAR-10), 6 output channels, 5x5 square convolution kernel
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
            # If the size is a square, you can specify with a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
            x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        def get_embeddings(self, x):
            # Extract embeddings from the second-to-last fully connected layer (fc2)
            x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
            x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            return x

    def evaluate_model(model, data_loader, criterion):
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

        loss = running_loss / len(data_loader)
        accuracy = 100. * correct / total

        return loss, accuracy

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    PATH = './cifar_resnet.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)


    # Assuming you have downloaded the CIFAR-10 dataset using torchvision.datasets.CIFAR10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)


    # Lists to store losses and accuracies
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Lists to store embeddings
    train_embeddings = []
    test_embeddings = []

    # Training loop
    num_epochs = 50

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
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
        #scheduler.step()

        # Evaluate on training and test set
        train_loss, train_accuracy = evaluate_model(net, train_loader, criterion)
        test_loss, test_accuracy = evaluate_model(net, test_loader, criterion)

        # Store losses and accuracies
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print statistics every epoch
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    train_embeddings = np.array(train_embeddings)
    np.save('train_embeddings.npy', train_embeddings)

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
    plt.show()
    plt.savefig('charts.png')

