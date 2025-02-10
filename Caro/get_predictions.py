import prepare_image_data
import torch
import torch.nn as nn
import get_labels
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

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

    def get_name(self):
        return "VGG-16"

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
model = MyNeuralNetwork()
model.load_state_dict(torch.load(project_root + "vgg16_model2.pth", map_location=torch.device('cpu')))

def test(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            all_labels.append(labels.numpy())

            inputs = inputs.to(device)

            # forward
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_predictions.append(preds.cpu())

            if batch_idx % 10 == 0:
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

    return all_predictions, all_labels

train_data, test_data, train_data_paths, test_data_paths = prepare_image_data.prepare_image_data()

test_labels = get_labels.get_data_labels(test_data_paths)
train_labels = get_labels.get_data_labels(train_data_paths)

test_labels_numeric = [artstyles_dict[label] for label in test_labels]
train_labels_numeric = [artstyles_dict[label] for label in train_labels]

test_data_tensor = []
for image in test_data:
    image_tensor2 = torch.from_numpy(np.asarray(image)).float()
    image_tensor2 = image_tensor2.permute(2, 0, 1)
    test_data_tensor.append(image_tensor2)
test_data_tensor = torch.stack(test_data_tensor)
test_data = TensorDataset(test_data_tensor, torch.tensor(test_labels_numeric))

train_data_tensor = []
for image in train_data:
    image_tensor1 = torch.from_numpy(image).float()
    image_tensor1 = image_tensor1.permute(2, 0, 1)
    train_data_tensor.append(image_tensor1)
train_data_tensor = torch.stack(train_data_tensor)
train_data = TensorDataset(train_data_tensor, torch.tensor(train_labels_numeric))

test_loader = DataLoader(dataset=test_data, shuffle=False)
train_loader = DataLoader(dataset=train_data, shuffle=False)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_predictions, all_labels = test(model, test_loader, device)
preds_list = [pred.item() for pred in all_predictions]
labels_list = [label.item() for label in all_labels]

df_testpred = pd.DataFrame({'TestPredictions': preds_list, 'True Values': labels_list})
df_testpred.to_csv("TestPredictions.csv", index=False)

all_predictions, all_labels = test(model, train_loader, device)
preds_list = [pred.item() for pred in all_predictions]
labels_list = [label.item() for label in all_labels]

df_testpred = pd.DataFrame({'TrainPredictions': preds_list, 'True Values': labels_list})
df_testpred.to_csv("TrainPredictions.csv", index=False)