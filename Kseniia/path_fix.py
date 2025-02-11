import os
import unicodedata
import re


def clean_filename(filename):
    """Fix encoding issues and keep only English characters."""
    # Fix common encoding issues
    replacements = {
        "ã¨": "e", "ã­": "i", "ã©": "e", "ã³": "o",
        "ã¶": "o", "ã¼": "u", "â\xa0": "a",
    }

    for wrong, correct in replacements.items():
        filename = filename.replace(wrong, correct)

    # Normalize Unicode characters (e.g., é → e, ö → o)
    filename = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("utf-8")

    # Remove any remaining non-alphanumeric characters (keep only A-Z, a-z, 0-9, _, and -)
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)

    return filename


def rename_files(directory):
    """Renames all files in a directory by fixing encoding issues."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            old_path = os.path.join(root, file)
            new_file = clean_filename(file)
            new_path = os.path.join(root, new_file)

            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Failed to rename {old_path}: {e}")


# Run it on your dataset folder
dataset_path= r"C:\Users\K-2\Desktop\Kseniia LFI\wikiart"
rename_files(dataset_path)
