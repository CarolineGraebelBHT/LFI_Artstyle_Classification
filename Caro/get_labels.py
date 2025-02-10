import re
import dataloader


def get_data_labels(path_list):
    print("Preparing labels...")
    artstyles = []
    for path in path_list:
        # match word after Data\ in path which holds the correct artstyle
        pattern = r"Data\\(.*?)\\"
        match = re.search(pattern, path)[1]
        if match:
            artstyle_found = match
        artstyles.append(artstyle_found)

    return artstyles
