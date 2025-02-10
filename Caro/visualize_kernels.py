import torch
import torch.nn as nn
import prepare_image_data
import dataloader
import get_labels
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from dataloader import train_test_split


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



project_root = "C:/Users/carol/Dropbox/DataScience/Semester3/Learning from Images/Project/LFI_Artstyle_Classification/Caro/"
model = MyNeuralNetwork()
model.load_state_dict(torch.load(project_root + "vgg16_model3.pth", map_location=torch.device('cpu')))
for name, module in model.named_modules():
    print(f"Name: {name}, Module: {module}")
model.eval()

artstyles_dict = {
    'Abstract_Expressionism': 0,
    'Baroque': 1,
    'Cubism': 2,
    'Expressionism': 3,
    'High_Renaissance': 4,
    'Impressionism': 5,
    'Realism': 6
}

target_layer_name = 'features.3'
target_layer = None

for name, module in model.named_modules():
    if name == target_layer_name:
        target_layer = module
        break

if target_layer == None:
    print("Target-Layer not found")
else:
    print("Target-Layer found")

def visualize_filters(layer, layer_name):
    print("Getting filter...")
    filters = layer.weight.data.cpu().numpy()  # Get filter weights, move to CPU, convert to NumPy
    n_filters, n_channels, _, _ = filters.shape

    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

    for i in range(n_filters):
        row = i // n_cols
        col = i % n_cols
        for c in range(n_channels): # Visualize each channel of the filter
            filter_img = filters[i, c, :, :]
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min()) # Normalize
            axes[row, col].imshow(filter_img, cmap='gray') # Most conv layers have gray scale filters
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    plt.suptitle(f"Filters of {layer_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

visualize_filters(target_layer, target_layer_name)