import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataload import load_preprocess_data
from ResNet import ResNet50
from torchvision import transforms
from PIL import Image
import random
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# Hyperparameters
# ===========================

batch_size = 64
num_epochs = 50
learning_rate = 0.001
momentum = 0.9

# Load dataset
data_dir = r"C:\Users\K-2\Desktop\Kseniia LFI\wikiart"
train_loader, val_loader, test_loader, class_names = load_preprocess_data(data_dir, batch_size=batch_size, quick_test=False)

# Set device (use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Model ResNet50
model = ResNet50(num_classes=len(class_names)).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Learning Rate Scheduler: Reduce LR by half every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training Variables
val_loss_history, val_acc_history = [], []
test_loss_history = []
test_acc_history = []
best_acc = 0.0
since = time.time()

early_stopping_patience = 10  # Stop if val loss doesn't improve for 10 epochs
best_val_loss = float('inf')
patience_counter = 0
# ===========================
# Training Loop
# ===========================
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print('-' * 50)

    # Training Phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"Training Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Validation Phase
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx}/{len(val_loader)} | Loss: {loss.item():.4f}")

    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_acc = 100 * val_correct / val_total
    val_loss_history.append(epoch_val_loss)
    val_acc_history.append(epoch_val_acc)

    print(f"Epoch {epoch+1} - Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")

    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), "saved_models/best_resnet_artstyle.pth")
        print("New Best Model Saved!")

    #Stop if val loss doesn't improve for 10 epochs
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0  # Reset patience
    else:
        patience_counter += 1  # Increase patience

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered. Training stopped.")
        break  # Stop training
    #---------------------------------------------------------

    scheduler.step()
    print(f"Learning Rate Updated: {scheduler.get_last_lr()[0]:.6f}")

# Training Summary
time_elapsed = time.time() - since
print(f"\nTraining Complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s")
print(f"Best Validation Accuracy: {best_acc:.2f}%")

# Load Best Model
best_model_path = "saved_models/best_resnet_artstyle.pth"
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("\nBest model loaded for evaluation & predictions.")
else:
    print("\nBest model not found! Using last trained model.")

# Evaluate on Test Set
def evaluate_on_test_set(model, test_loader):
    """
    Evaluates the model on the entire test dataset, printing batch-wise progress.
    """
    model.eval()
    test_running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                batch_acc = 100 * correct / total
                print(f"Test Batch {batch_idx}/{len(test_loader)} | Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%")

    # Compute final metrics
    test_loss = test_running_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    # Append to history for plotting
    test_loss_history.append(test_loss)
    test_acc_history.append(test_accuracy)

    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# Run evaluation
evaluate_on_test_set(model, test_loader)

# ===========================
# Plot Validation & Test Loss / Accuracy
# ===========================

def plot_results(val_data, test_data, title, ylabel, filename):
    """
    Plots validation loss/accuracy across epochs and final test loss/accuracy.
    """
    if not val_data:
        print(f"Skipping {title} plot due to missing validation data.")
        return
    if not test_data:
        print(f"Skipping {title} plot due to missing test data.")
        return

    epochs_range = range(1, len(val_data) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, val_data, label="Validation", marker='o')

    #Only plot test data if it exists
    if test_data:
        plt.scatter(num_epochs, test_data[0], color='red', label="Final Test Value")  # Single test point

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()

    # Save the plot
    os.makedirs("results", exist_ok=True)
    filepath = f"results/{filename}.png"
    plt.savefig(filepath, bbox_inches="tight")
    print(f"\nPlot saved: {filepath}")

    plt.show()


# Save Validation & Test Plots
print("Test Loss History:", test_loss_history)
print("Test Accuracy History:", test_acc_history)

plot_results(val_loss_history, test_loss_history, "Validation & Test Loss", "Loss", "val_test_loss_plot")
plot_results(val_acc_history, test_acc_history, "Validation & Test Accuracy", "Accuracy", "val_test_accuracy_plot")


# ===========================
# Predict & Visualize 6 Random Test Images
# ===========================
def extract_real_label(image_path, class_names):
    """
    Extracts the real class label from the image path (based on folder structure).
    """
    real_class_name = os.path.basename(os.path.dirname(image_path))  # Get folder name
    if real_class_name in class_names:
        return real_class_name
    else:
        return "Unknown"  # In case of missing/mislabeled files

def predict_multiple_images(model, image_paths, class_names, num_rows=2, num_cols=3, save_path="results/test_predictions.png"):
    """
    Predicts multiple images, plots them in a grid with their real and predicted labels, and saves the visualized predictions.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        transformed_image = transform(image).unsqueeze(0).to(device)

        # Get predicted class
        with torch.no_grad():
            output = model(transformed_image)
            _, predicted_class = torch.max(output, 1)

        predicted_label = class_names[predicted_class.item()]
        real_label = extract_real_label(image_path, class_names)

        # Plot Image
        ax = axes[idx // num_cols, idx % num_cols]
        ax.imshow(image)
        ax.set_title(f"Real: {real_label}\nPredicted: {predicted_label}", fontsize=10, color="green" if real_label == predicted_label else "red")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nPredictions saved at: {save_path}")
    plt.show()

# Select & Predict Random Images from Test Dataset
test_image_paths = glob.glob(r"C:\Users\K-2\Desktop\Kseniia LFI\wikiart\*/*.jpg")
test_images = random.sample(test_image_paths, 6) if len(test_image_paths) >= 6 else test_image_paths
predict_multiple_images(model, test_images, class_names)
