import re
import dataloader


def get_data_labels(path_list):
    """This function takes image paths and uses a regex match to extract the correct art style."""
    print("Preparing labels...")
    artstyles = []
    for path in path_list:
        # match word after Data\ in path which holds the correct art style
        # for dataloader.py
        pattern = r"Data\\(.*?)\\"

        # pattern for when using dataloader2
        #pattern = r"Data/(.*?)\\"

        match = re.search(pattern, path)[1]
        if match:
            artstyle_found = match
        artstyles.append(artstyle_found)

    return artstyles
