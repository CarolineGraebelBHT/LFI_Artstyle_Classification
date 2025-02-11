import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# data_dir = r"C:\Users\K-2\Desktop\Kseniia LFI\wikiart"

# Define class labels
CATEGORIES = {
    0: 'Baroque', 1: 'Impressionism', 2: 'Post_Impressionism',
    3: 'Abstract_Expressionism', 4: 'Analytical_Cubism',
    5: 'Cubism', 6: 'Synthetic_Cubism', 7: 'Realism',
    8: 'New_Realism', 9: 'Contemporary_Realism',
    10: 'Early_Renaissance', 11: 'Mannerism_Late_Renaissance',
    12: 'Northern_Renaissance', 13: 'High_Renaissance'
}

def load_preprocess_data(data_dir, batch_size, quick_test=False):
    """
    Loads dataset from folders, splits into Train (70%), Validation (15%), Test (15%),
    and returns DataLoaders.
    """
    IMG_SIZE = (256, 256)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Flip randomly
        transforms.RandomRotation(15),  # Small rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Splitting dataset: 70% Train, 15% Validation, 15% Test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # ===================
    # If quick_test=True, use a smaller dataset
    if quick_test:
        train_dataset = Subset(train_dataset, range(3000))  # Use only 100 images for training
        val_dataset = Subset(val_dataset, range(500))  # Use only 50 images for validation
    # =====================

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data loaded successfully! Total images: {len(dataset)}")
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes: {len(dataset.classes)} -> {dataset.classes}")

    return train_loader, val_loader, test_loader, dataset.classes