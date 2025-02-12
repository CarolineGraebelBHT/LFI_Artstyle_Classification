import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
import get_labels
import time
import prepare_image_data
import os

artstyles_dict = {
    'Abstract_Expressionism': 0,
    'Baroque': 1,
    'Cubism': 2,
    'Expressionism': 3,
    'High_Renaissance': 4,
    'Impressionism': 5,
    'Realism': 6
}

project_root = "C:/Users/carol/Dropbox/DataScience/Semester3/Learning from Images/Project/LFI_Artstyle_Classification/Caro/"
model_save_path = os.path.join(project_root, "vgg16_model.pth")

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # source: https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918
        self.features = nn.Sequential(
            # first stage
            nn.Conv2d(3, 64, kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # second stage
            nn.Conv2d(64, 128, kernel_size=3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # third stage
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # fourth stage
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # fifth stage
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def name(self):
        return "VGG-16"

def training(model, data_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # backward
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if batch_idx % 10 == 0:
            print(f'Training Batch: {batch_idx:4} of {len(data_loader)}')

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc

def test(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc

#print(torch.version.cuda)
#print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

# set seed for reproducability
torch.manual_seed(0)

# hyperparameters
batch_size = 32
num_epochs = 20
learning_rate =  0.001
momentum = 0.9

def plot(train_history, test_history, metric, num_epochs):

    plt.title(f"Validation/Test {metric} vs. Number of Training Epochs VGG-16")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Test {metric}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), test_history, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{metric}_VGG-16.png")
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    # load train and test data
    train_data, test_data, train_data_paths, test_data_paths = prepare_image_data.prepare_image_data()

    train_labels = get_labels.get_data_labels(train_data_paths)
    test_labels = get_labels.get_data_labels(test_data_paths)

    train_labels_numeric = [artstyles_dict[label] for label in train_labels]
    test_labels_numeric = [artstyles_dict[label] for label in test_labels]

    train_data_tensor = []
    for image in train_data:
        image_tensor1 = torch.from_numpy(image).float()  # Convert to tensor
        image_tensor1 = image_tensor1.permute(2, 0, 1)
        train_data_tensor.append(image_tensor1)  # Append the tensor to the list
    train_data_tensor = torch.stack(train_data_tensor)  # Stack the tensors after converting all images
    train_data = TensorDataset(train_data_tensor, torch.tensor(train_labels_numeric))

    test_data_tensor = []
    for image in test_data:
        image_tensor2 = torch.from_numpy(np.asarray(image)).float()
        image_tensor2 = image_tensor2.permute(2, 0, 1)
        test_data_tensor.append(image_tensor2)
    test_data_tensor = torch.stack(test_data_tensor)
    test_data = TensorDataset(test_data_tensor, torch.tensor(test_labels_numeric))

    loader_params = {
        'batch_size': batch_size,
        'num_workers': 5
    }

    train_loader = DataLoader(dataset=train_data, shuffle=False, **loader_params)
    test_loader = DataLoader(dataset=test_data, shuffle=False, **loader_params)

    model = MyNeuralNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_acc_history = []
    test_acc_history = []

    train_loss_history = []
    test_loss_history = []

    best_acc = 0.0
    since = time.time()

    print("Starting the training.")
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train
        training_loss, training_acc = training(model, train_loader, optimizer,
                                               criterion, device)
        train_loss_history.append(training_loss)
        train_acc_history.append(training_acc)

        # test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # overall best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)

    time_elapsed = time.time() - since
    print(
        f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'
    )
    print(f'Best val Acc: {best_acc:4f}')

    # plot loss and accuracy curves
    train_acc_history = [h.cpu().numpy() for h in train_acc_history]
    test_acc_history = [h.cpu().numpy() for h in test_acc_history]
    plot(train_acc_history, test_acc_history, 'accuracy', num_epochs)
    plot(train_loss_history, test_loss_history, 'loss', num_epochs)