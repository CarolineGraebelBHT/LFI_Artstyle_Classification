import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from ResNet import ResNet50
from dataload import CATEGORIES, load_preprocess_data
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== Correct class order ====================
# Sort classes alphabetically to match training label order
class_names = sorted(CATEGORIES.values())

# ==================== Proper device setup ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== Model Loading ====================
model = ResNet50(num_classes=len(class_names)).to(device)

best_model_path = "saved_models/best_resnet_artstyle.pth"
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print("\nBest model loaded successfully!")
else:
    raise FileNotFoundError("No saved model found!")

# ==================== Feature Map Extraction ====================
def get_transform():
    """Must match validation/test transforms from training"""
    return transforms.Compose([
        transforms.Resize(256),          # Match training resize
        transforms.CenterCrop(224),      # Use center crop for evaluation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Keep consistent with training
    ])

# ==================== Load Test Data ====================
data_dir = r"D:\kaggle\wikiart\wikiart"  # Ensure correct dataset path
batch_size = 32
_, _, test_loader, _ = load_preprocess_data(data_dir, batch_size=batch_size, quick_test=False)  # Use actual DataLoader

#==================== Confusion Matrix Function ====================
def plot_normalized_confusion_matrix(model, dataloader, class_names):
    """
    Generates a normalized confusion matrix where each row sums to 1 (percentage format).
    This makes the matrix more interpretable for imbalanced datasets.
    """
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    # Normalize by row (divide each row by the total true count for that class)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix (Percentage per Class)")
    plt.show()

# Call the function to plot
plot_normalized_confusion_matrix(model, test_loader, class_names)
