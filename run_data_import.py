import detection.dataset as dataset

if __name__ == '__main__':
    folder = '../../Radardata/2018-05-30'
    time_step = 5
    dataset.split_folder(folder, time_step)