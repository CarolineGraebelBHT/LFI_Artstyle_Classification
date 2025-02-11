import os
import torch
import cv2
import concurrent.futures
from dataload import load_preprocess_data

def get_image_paths(data_loader, limit=None):
    """
    Extracts image file paths from the dataset.
    Handles both `ImageFolder` and `Subset` datasets.
    Allows limiting the number of images for quick testing.
    """
    dataset = data_loader.dataset

    # Unwrap `Subset` dataset if needed
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    # Extract file paths if dataset is `ImageFolder`
    if hasattr(dataset, "samples"):
        paths = [sample[0] for sample in dataset.samples]
        return paths[:limit] if limit else paths
    else:
        print("Dataset does not contain image file paths.")
        return []

def check_image(path):
    """
    Uses OpenCV to check if an image is readable.
    Prints errors but does not store paths.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Corrupt image: {path}")

def find_faulty_data_paths(image_paths):
    """
    Uses multithreading to speed up corrupt image detection.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(check_image, image_paths)

# 🔹 Load dataset without modifying `dataload.py`
data_dir = r"C:\Users\K-2\Desktop\Kseniia LFI\wikiart"
batch_size = 32

train_loader, val_loader, test_loader, _ = load_preprocess_data(data_dir, batch_size=batch_size, quick_test=True)

# 🔹 Extract image paths (limit to 1500 for quick testing)
N = 2500
train_data_paths = get_image_paths(train_loader, limit=N)
val_data_paths = get_image_paths(val_loader, limit=N)
test_data_paths = get_image_paths(test_loader, limit=N)

# 🔹 Check for corrupt images (faster with threading)
print("\nChecking training dataset (Limited to 1500 images)...")
find_faulty_data_paths(train_data_paths)

print("\nChecking validation dataset (Limited to 1500 images)...")
find_faulty_data_paths(val_data_paths)

print("\nChecking testing dataset (Limited to 1500 images)...")
find_faulty_data_paths(test_data_paths)

print("\nImage corruption check complete. No files were deleted.")
