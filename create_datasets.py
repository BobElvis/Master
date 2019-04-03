import dataset as dataset
import os
import os.path as path
from datetime import datetime


if __name__ == '__main__':
    root = dataset.ROOT_FOLDER
    folders = dataset.find_folders(root)
    for folder in folders:
        dataset.validate_folder(path.join())
