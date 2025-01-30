import glob
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random

def load_image_paths():
    os.chdir("..")
    print(f"Current working directory: {os.getcwd()}")

    image_paths = glob.glob('./Data/*/*.jpg')

    return image_paths

def train_test_split(image_paths, train_ratio=0.7, seed=42):
    n = len(image_paths)
    train_size = int(n * train_ratio)

    random.seed(seed)
    random.shuffle(image_paths)

    train_list = image_paths[:train_size]
    test_list = image_paths[train_size:]

    return train_list, test_list

def prep_train_test_data():
    image_paths = load_image_paths()

    train_data_paths, test_data_paths = train_test_split(image_paths)

    return train_data_paths, test_data_paths

#train, test = prep_train_test_data()