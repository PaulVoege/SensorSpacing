import glob
import os
import sys
import re

def get_number_from_filename(filename):
    match = re.search(r'net_epoch(\d+).pth', filename)
    return int(match.group(1)) if match else 0

def clean_all_numeric_checkpoint(path):
    """
    remove all intermediate checkpoint except bestval

    path: str,
        a path to log directory
    """
    file_list = glob.glob(os.path.join(path, "net_epoch[0-9]*.pth"))
    file_list = sorted(file_list, key=get_number_from_filename)
    for file in file_list[1:-1]:
        os.remove(file)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.isdir(path)
    clean_all_numeric_checkpoint(path)