import re


def get_data_labels(path_list):
    artstyles = []
    for path in path_list:
        pattern = r"Data\\(.*?)\\"

        match = re.search(pattern, path)[1]
        if match:
            artstyle_found = match  # Store the matched label
        artstyles.append(artstyle_found)  # Add the found label or None if not found to the result list

    return artstyles
