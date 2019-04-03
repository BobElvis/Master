class DataConfig:
    def __init__(self, radar_range, dt):
        self.radar_range = radar_range
        self.dt = dt
        self.rotation = ROTATION
        self.radar_img_size = RADAR_IMG_SIZE

    def __repr__(self):
        return "(range: {}, rotation: {}, dt: {})"\
            .format(self.radar_range, self.rotation, self.dt)


def get_data_config(folder):
    # Returns most specific data config.
    day_config = _data_configs.get(folder)
    if day_config is None:
        print("WARNING: Config for {} not found. Returning default".format(folder))
        config = _data_configs.get(None)
    else:
        config = day_config
    return DataConfig(config[0], config[1])


def add_data_config(folder, radar_range_setting, dt):
    _data_configs[folder] = (int(round(radar_range_setting*RADAR_SCALE)), dt)


_data_configs = {}

# CONSTANTS:
RADAR_IMG_SIZE = 1024
ROTATION = 183
RADAR_SCALE = 200/175  # actual range / radar setting

# SET CONFIGS HERE:
add_data_config(None, 175, 5)
add_data_config('2018-05-28', 175, 1.25)
add_data_config('2018-05-30', 175, 5)
