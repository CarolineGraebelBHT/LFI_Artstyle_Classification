import os
import json
import matplotlib.pyplot as plt

# Load the saved history file
history_file = "saved_models/training_history.json"

if not os.path.exists(history_file):
    print("Error: Training history file not found!")
    exit()

# Load training data
with open(history_file, "r") as f:
    history = json.load(f)

# Extract validation loss and accuracy from epochs
val_loss_history = [epoch["val_loss"] for epoch in history["epochs"]]
val_acc_history = [epoch["val_accuracy"] for epoch in history["epochs"]]

# Extract test loss and accuracy as single values
test_loss = history["test_loss"]
test_accuracy = history["test_accuracy"]

epochs_range = range(1, len(val_loss_history) + 1)  # Match actual trained epochs (41)

def plot_results(val_data, test_value, title, ylabel, filename):
    """
    Plots validation loss/accuracy across trained epochs and adds final test result.
    """
    if not val_data:
        print(f"Skipping {title} plot due to missing validation data.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, val_data, label="Validation", marker='o')

    # Plot test data at the last trained epoch (instead of epoch 50)
    plt.scatter(len(val_loss_history), test_value, color='red', label="Final Test Value")  # Single test point

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

# Regenerate the plots correctly
plot_results(val_loss_history, test_loss, "Validation & Test Loss", "Loss", "val_test_loss_plot")
plot_results(val_acc_history, test_accuracy, "Validation & Test Accuracy", "Accuracy", "val_test_accuracy_plot")

