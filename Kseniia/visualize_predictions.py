import os
import glob
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from ResNet import ResNet50
from dataload import CATEGORIES

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== Correct class order ====================
# Sort classes alphabetically to match training label order
class_names = sorted(CATEGORIES.values())  # Critical fix!

# ==================== Proper device setup ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== Model loading ====================
model = ResNet50(num_classes=len(class_names)).to(device)

# Load weights properly
best_model_path = "saved_models/best_resnet_artstyle.pth"
if os.path.exists(best_model_path):
    # Remove weights_only=True and use strict=True
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True), strict=True)
    model.eval()
    print("\nBest model loaded successfully!")
else:
    raise FileNotFoundError("No saved model found!")

# ====================Correct transforms ====================
def get_transform():
    """Must match validation/test transforms from training"""
    return transforms.Compose([
        transforms.Resize(256),          # Match training resize
        transforms.CenterCrop(224),      # Use center crop for evaluation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Keep consistent with training
    ])

# ==================== Prediction function ====================
def predict_and_visualize(image_paths, num_images=6):
    transform = get_transform()
    selected_paths = random.sample(image_paths, num_images)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, path in enumerate(selected_paths):
        try:
            # Load and transform image
            image = Image.open(path).convert("RGB")
            tensor_img = transform(image).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(tensor_img)
                _, pred_idx = torch.max(outputs, 1)
                predicted = class_names[pred_idx.item()]

            # Get true label from path
            true_label = os.path.basename(os.path.dirname(path))

            # Plot
            ax = axes[idx // 3, idx % 3]
            ax.imshow(image)
            ax.set_title(f"True: {true_label}\nPred: {predicted}",
                         color="green" if true_label == predicted else "red")
            ax.axis("off")

        except Exception as e:
            print(f"Error processing {path}: {str(e)}")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = "results/random_predictions.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nPredictions saved at: {save_path}")
    plt.show()


# ==================== Get all images ====================
all_images = glob.glob(r"C:\Users\K-2\Desktop\Kseniia LFI\wikiart\*/*.jpg")

# ==================== Run prediction ====================
if not all_images:
    print("No images found!")
    exit()

predict_and_visualize(all_images)