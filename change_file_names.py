import dataset as dataset
import os
import os.path as path
from datetime import datetime


if __name__ == '__main__':
    accept_error = True
    format_string = dataset.format_str

    root = dataset.ROOT_FOLDER + "/"
    folders = dataset.find_folders(root)
    for folder in folders:
        folder_path = path.join(root, folder)
        files = dataset.find_files(folder_path, None)
        new_name = [folder + " " + f for f in files]

        old_name_valid = []
        new_name_valid = []

        # Check if new names are valid:
        for new_name, old_name in zip(new_name, files):
            try:
                date = datetime.strptime(path.splitext(new_name)[0], format_string)
            except ValueError:
                if accept_error:
                    continue
                else:
                    print("{} -> {}".format(old_name, new_name))
                    raise ValueError
            old_name_valid.append(old_name)
            new_name_valid.append(new_name)

        # Rename files:
        old_file_name = [path.join(folder_path, f) for f in old_name_valid]
        new_file_name = [path.join(folder_path, f) for f in new_name_valid]

        for i, (o_f, n_f) in enumerate(zip(old_file_name, new_file_name)):
            os.rename(o_f, n_f)