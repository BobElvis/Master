import os
from datetime import datetime
import detection.dataconfigs

ROOT_FOLDER = "../../Radardata"
ROOT_FOLDER_SAVE = "../../MHTData/MHTSaves"
ROOT_FOLDER_TRUTH = "../../MHTData/TruthSaves"
format_str = "%Y-%m-%d %H_%M_%S"
radar_suff = ".png"
camera_suff = ".jpg"

# Basic:


def find_folders(folder):
    folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    folders.sort()
    return folders


def find_files(folder, extension=radar_suff):
    if extension is not None:
        file_names = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(extension) and not os.path.isdir(f)]
    else:
        file_names = [f for f in os.listdir(folder) if not os.path.isdir(f)]
    file_names.sort()
    return file_names


def filename_to_date(file_name):
    return datetime.strptime(file_name, format_str)

# Complex:


def validate_folder(dir, dt):
    file_names = find_files(dir)
    time_str_iter = enumerate(file_names)
    time_prev = filename_to_date(time_str_iter.__next__()[1])
    for idx, time_str in time_str_iter:
        time_now = filename_to_date(time_str)
        diff = time_now - time_prev
        if diff.seconds > dt:
            print(" i={}/{}. {}. {}".format(idx, len(file_names), time_str, diff.seconds))
        time_prev = time_now


def split_folder(dir, dt):
    partitions = []

    # 1. Find files:
    file_names = find_files(dir, extension=radar_suff)
    if len(file_names) == 0:
        raise FileNotFoundError("No files found in folder.")

    # 2. Determine partitions:
    iterator = enumerate(file_names)
    first_file = iterator.__next__()[1]
    time_prev = datetime.strptime(first_file, format_str)
    current_partition = [first_file]
    partitions.append(current_partition)

    for idx, file in iterator:
        time_now = datetime.strptime(file, format_str)
        if (time_now - time_prev).seconds > dt:
            # Create new partition:
            current_partition = [file]
            partitions.append(current_partition)
        else:
            current_partition.append(file)
        time_prev = time_now

    print("Number of partitions: {}".format(len(partitions)))

    # 3. Create folders:
    new_directories = []
    for idx, partition in enumerate(partitions):
        print(" -Partition {}: {} elements".format(idx, len(partition)))
        new_dir = os.path.join(dir, str(idx))
        if os.path.exists(new_dir):
            return " -ERROR. Subfolder {} exists...".format(idx)
        else:
            os.makedirs(new_dir)
        new_directories.append(new_dir)

    # 4. Move files:
    for idx, partition in enumerate(partitions):
        new_dir = new_directories[idx]
        for idx2, file in enumerate(partition):
            r_name = file + radar_suff
            c_name = file + camera_suff

            old_radar = os.path.join(dir, r_name)
            old_camera = os.path.join(dir, c_name)
            new_radar = os.path.join(new_dir, r_name)
            new_camera = os.path.join(new_dir, c_name)

            os.rename(old_radar, new_radar)

            # Sometimes camera is not loaded:
            try:
                os.rename(old_camera, new_camera)
            except FileNotFoundError:
                print(" -No camera for {}".format(file))


class Dataset:
    def __init__(self, config):
        self.config = config

    def __getitem__(self, item):
        return None

    def __len__(self):
        return None

    def __str__(self):
        return None


class DatasetFolder(Dataset):
    def __init__(self, root: str, folder: str):
        super().__init__(detection.dataconfigs.get_data_config(folder))
        self.folder_name = folder
        self.dir = os.path.join(root, folder)
        self.files = find_files(self.dir)

    def __getitem__(self, index):
        return os.path.join(self.dir, self.files[index])

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "n: {}, dir: {}, config: {}".format(len(self), self.dir, self.config)

    def __str__(self):
        return "{}".format(self.folder_name)
