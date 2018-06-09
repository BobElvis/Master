class DataConfig:
    def __init__(self, radar_range, dt):
        self.radar_range = radar_range
        self.dt = dt
        self.rotation = ROTATION
        self.radar_img_size = RADAR_IMG_SIZE

    def __repr__(self):
        return "(range: {}, rotation: {}, dt: {})"\
            .format(self.radar_range, self.rotation, self.dt)


def get_data_config(day=None, partition=None):
    # Returns most specific data config.
    day_configs = _data_configs.get(day)
    if day_configs is None:
        print("WARNING: Config for {}:{} not found. Returning default".format(day, partition))
        config = _data_configs.get(None)
    else:
        partition_config = day_configs.get(partition)
        if partition_config is None:
            config = day_configs.get(None)
        else:
            config = partition_config
    return DataConfig(config[0], config[1])


def add_data_config(day, partition, radar_range_setting, dt):
    day_configs = _data_configs.setdefault(day, {})
    day_configs[partition] = (int(round(radar_range_setting*RADAR_SCALE)), dt)


_data_configs = {}

# CONSTANTS:
RADAR_IMG_SIZE = 1024
ROTATION = 183
RADAR_SCALE = 200/175  # actual range / radar setting

# SET CONFIGS HERE:
add_data_config(None, None, 175, 5)
add_data_config('2018-05-28', None, 175, 1.25)
add_data_config('2018-05-30', None, 175, 5)
