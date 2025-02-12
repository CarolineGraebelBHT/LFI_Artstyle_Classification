import glob
import random

def load_image_paths():
    """This function extracts all image paths from the project folder Data. Important is that only the art styles
    of interest are present in the Data folder, which is only a subset of the full Wikiart set."""

    # for this, only relevant genre folders need to remain, rest gets deleted
    image_paths = glob.glob('.././Data/*/*.jpg')

    return image_paths

def train_test_split(image_paths, train_ratio=0.7, seed=42):
    """This function shuffles the path list, extracts the first 10000 image paths and splits the list into training
    and testing data paths depending on the given train_ratio."""
    random.seed(seed)
    random.shuffle(image_paths)
    image_paths = image_paths[0:10000]
    n = len(image_paths)
    train_size = int(n * train_ratio)

    train_list = image_paths[:train_size]
    test_list = image_paths[train_size:]

    return train_list, test_list

def prep_train_test_data():
    """This function loads the paths to the images in the Data folder and splits the resulting list into training
    and testing data paths."""
    image_paths = load_image_paths()

    train_data_paths, test_data_paths = train_test_split(image_paths)

    return train_data_paths, test_data_paths