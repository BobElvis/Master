import detection.dataset as dataset

if __name__ == '__main__':
    folder = '../../Radardata/2018-05-28'
    time_step = 1.25
    dataset.split_folder(folder, 1.25)