import dataloader
import get_labels
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

"""This script reads the labels from the train and testing path data and plots their distribution as a pie chart for
training and testing respectively."""

train_paths, test_paths = dataloader.prep_train_test_data()
train_labels = get_labels.get_data_labels(train_paths)
test_labels = get_labels.get_data_labels(test_paths)

unique_labels, counts = np.unique(train_labels, return_counts=True)

plt.figure(figsize=(8, 8))
plt.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Art Categories in Training Data")
plt.axis('equal')
plt.show()

unique_labels, counts = np.unique(test_labels, return_counts=True)
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Art Categories in Testing Data")
plt.axis('equal')
plt.show()

default_colors = rcParams['axes.prop_cycle'].by_key()['color'][0:7]
print(default_colors)

color_map = dict(zip(unique_labels, default_colors)) # mapping the labels to their colors
print(color_map)