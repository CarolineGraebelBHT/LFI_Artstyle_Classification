import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
from ResNet import ResNet50
from dataload import CATEGORIES, load_preprocess_data

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

# ==================== Class-Wise Accuracy Calculation ====================
def class_wise_accuracy(model, dataloader, class_names):
    model.eval()
    correct = np.zeros(len(class_names))
    total = np.zeros(len(class_names))

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()
                total[label] += 1
                correct[label] += (preds[i] == labels[i]).item()

    # Avoid division by zero for classes with no test samples
    total = np.where(total == 0, 1, total)

    acc_per_class = 100 * (correct / total)

    print("\nClass-wise Accuracy:")
    for i, acc in enumerate(acc_per_class):
        print(f"{class_names[i]:<25}: {acc:.2f}% accuracy")

    return acc_per_class  # Fix: Return the computed accuracy scores


# ==================== Run Class-wise Accuracy Evaluation ====================
accuracy_scores = class_wise_accuracy(model, test_loader, class_names)

# ==================== Plot Class-wise Accuracy with Pastel Colors ====================
plt.figure(figsize=(12, 6))
pastel_colors = ['#e78ac3', '#8da0cb', '#fc8d62', '#66c2a5', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
                 '#ff69b4', '#d95f02', '#7570b3', '#1b9e77', '#d73027', '#66a61e']  # Same colors as previous bar plot

plt.bar(class_names, accuracy_scores, color=pastel_colors)

plt.xlabel("Art Style", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Class-wise Accuracy for ResNet-50", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()