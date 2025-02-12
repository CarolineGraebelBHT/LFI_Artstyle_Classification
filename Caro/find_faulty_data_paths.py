import dataloader
import cv2

def find_faulty_data_paths(data_paths):
    """This function tries to read in the images of each image paths and attaches it to an error list if it fails."""
    faulty_paths = []
    for path in data_paths:
        img = cv2.imread(path)
        if img is None:
            faulty_paths.append(path)
    return faulty_paths

# not used, only note for replacement
def patch_filename(f):
    """This function serves as a dictionary to the encoding errors in the faulty data paths. It is not possible
    to change or otherwise access the faulty file paths through Python at all."""
    f = f.replace("ã¨", "e").replace("ã­", "í").replace("ã©", "e").replace("ã³", "o")
    f = f.replace("ã¶", "oe").replace("ã¼", "ue").replace("â\xa0", "a")
    return f


train_data_paths, test_data_paths = dataloader.prep_train_test_data()

# print faulty paths and amount of faulty paths
faulty_paths1 = find_faulty_data_paths(train_data_paths)
print(f"Faulty paths training data: {faulty_paths1}")
print(f"Amount of wrong paths in the training data: {len(faulty_paths1)}")

faulty_paths2 = find_faulty_data_paths(test_data_paths)
print(f"Faulty paths testing data: {faulty_paths2}")
print(f"Amount of wrong paths in the testing data: {len(faulty_paths2)}")

