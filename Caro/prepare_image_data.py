import cv2
import dataloader
import numpy as np

def crop_to_largest_square(img):
    height, width, _ = img.shape

    # prepare biggest possible square out of the image by pixels
    square_size = min(width, height)

    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size

    cropped_img = img[top:bottom, left:right]

    return cropped_img

def prepare_image_data():
    print("Loading image paths...")
    train_data_paths, test_data_paths = dataloader.prep_train_test_data()
    train_data = []
    test_data = []

    print("Preparing train data...")
    for path in train_data_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image {path}")
        else:
            img = crop_to_largest_square(img)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            train_data.append(img)

    print("Preparing test data...")
    for path in test_data_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image {path}")
        else:
            img = crop_to_largest_square(img)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            test_data.append(img)

    return train_data, test_data, train_data_paths, test_data_paths

#train_data, test_data, train_data_paths, test_data_paths = prepare_image_data()
#cv2.imshow("Image train", train_data[0])
#cv2.imshow("Image test", test_data[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()