import carla

DATASET_PATH = '/home/kishky/master/carla-journal/dataset/'
#DATASET_PATH = '/media/kishky/My Passport/Master/carla_research/dataset/'

#BRANCHES_DATASET_PATH = '/media/kishky/My Passport/Master/carla_research/dataset_branches/'
BRANCHES_DATASET_PATH = '/home/kishky/master/carla-journal/dataset_branches/'

EPISODE_SIMULATION_DURATION_SEC = 600    # 10 minutes long
EPISODE_START                   = 0
EPISODE_COUNT                   = 250    # number of episodes to be simulated

TRAFFIC_DENISTY_MIN             = 30
TRAFFIC_DENISTY_MAX             = 60

PEDESTRIAN_DESNISTY_MIN         = 50
PEDESTRIAN_DESNISTY_MAX         = 100

IMAGE_X                         = 800
IMAGE_Y                         = 600
IMAGE_CUT                       = [100, 500]

IMAGE_RESIZE_X                  = 200
IMAGE_RESIZE_Y                  = 88

LIDAR_CHANNELS                  = 32
LIDAR_RANGE                     = 15
LIDAR_FOV_U                     = 10
LIDAR_FOV_L                     = -30
LIDAR_POINTS_PER_SEC            = 150000
LIDAR_ROTATION_FREQ             = 10

SENSOR_TICK                     = 0.1   # 10 FPS
SIM_TICK                        = 0.05

# Steering wheel noise
APLLY_STEERING_NOISE_EVERY_SEC = 1
APPLY_NOISE_PROBABILITY =  0.4

APPLY_NOISE_MIN_T = 0.25
APPLY_NOISE_MAX_T = 0.5
NOISE_INTENSITY_MIN = 0.05
NOISE_INTENSITY_MAX = 0.10


PREDEFINED_WEATHER_LIST = [
carla.WeatherParameters.ClearNoon,
carla.WeatherParameters.WetNoon,
carla.WeatherParameters.HardRainNoon,
carla.WeatherParameters.ClearSunset,
]

SIMULATION_TOWN_LIST = ['Town01']
