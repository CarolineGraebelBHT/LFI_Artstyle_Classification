import glob
import random

def shuffle_paths(path_list, seed = 42):
    """This function shuffles the paths with a fixed seed for reproducibility."""
    random.seed(seed)
    random.shuffle(path_list)
    return path_list

def load_image_paths_per_category():
    """This function reads the images from each art folder respectively, shuffles them and selects the first 1000."""
    # for this, only relevant genre folders need to remain, rest gets deleted
    impressionism = shuffle_paths(glob.glob('.././Data/Impressionism/*.jpg'))[:1000]
    cubism = shuffle_paths(glob.glob('.././Data/Cubism/*.jpg'))[:1000]
    abstract_expressionism = shuffle_paths(glob.glob('.././Data/Abstract_Expressionism/*.jpg'))[:1000]
    baroque = shuffle_paths(glob.glob('.././Data/Baroque/*.jpg'))[:1000]
    high_renaissance = shuffle_paths(glob.glob('.././Data/High_Renaissance/*.jpg'))[:1000]
    realism = shuffle_paths(glob.glob('.././Data/Realism/*.jpg'))[:1000]
    expressionism = shuffle_paths(glob.glob('.././Data/Expressionism/*.jpg'))[:1000]

    return impressionism, cubism, abstract_expressionism, baroque, high_renaissance, realism,expressionism

def train_test_split(image_paths, train_ratio=0.7):
    """This function splits a path list into a training and a testing path list."""
    n = len(image_paths)
    train_size = int(n * train_ratio)

    train_list = image_paths[:train_size]
    test_list = image_paths[train_size:]

    return train_list, test_list

def prep_train_test_data():
    """This function prepares the training and testing data sets for each art style and then fuses the result into a big
    common training and testing data set. The resulting sets get shuffled and returned."""
    impressionism, cubism, abstract_expressionism, baroque, high_renaissance, realism, expressionism = load_image_paths_per_category()
    impressionism_train, impressionism_test = train_test_split(impressionism)
    cubism_train, cubism_test = train_test_split(cubism)
    abstract_expressionism_train, abstract_expressionism_test = train_test_split(abstract_expressionism)
    baroque_train, baroque_test = train_test_split(baroque)
    high_renaissance_train, high_renaissance_test = train_test_split(high_renaissance)
    realism_train, realism_test = train_test_split(realism)
    expressionism_train, expressionism_test = train_test_split(expressionism)

    train_data_paths = shuffle_paths(impressionism_train + cubism_train + abstract_expressionism_train + baroque_train + high_renaissance_train + realism_train + expressionism_train)
    test_data_paths = shuffle_paths(impressionism_test + cubism_test + abstract_expressionism_test + baroque_test + high_renaissance_test + realism_test + expressionism_test)

    return train_data_paths, test_data_paths