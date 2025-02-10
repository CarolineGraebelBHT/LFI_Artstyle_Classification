import glob
import random

def load_image_paths():

    # for this, only relevant genre folders need to remain, rest gets deleted
    image_paths = glob.glob('.././Data/*/*.jpg')

    return image_paths

def train_test_split(image_paths, train_ratio=0.7, seed=42):
    random.seed(seed)
    random.shuffle(image_paths)
    image_paths = image_paths[0:10000]
    n = len(image_paths)
    train_size = int(n * train_ratio)

    train_list = image_paths[:train_size]
    test_list = image_paths[train_size:]

    return train_list, test_list

def prep_train_test_data():
    image_paths = load_image_paths()

    train_data_paths, test_data_paths = train_test_split(image_paths)

    return train_data_paths, test_data_paths