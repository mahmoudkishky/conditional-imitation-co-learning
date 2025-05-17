import carla
from agents.navigation.local_planner import RoadOption
import random
import threading
import numpy as np
import logging
import time
import os
import cv2
import csv
from queue import Queue
from queue import Empty
import configs
from noiser import Noiser
from agents.navigation.basic_agent import BasicAgent
import math 

class dataCollector:
    def __init__(self):

        self.currentEpisodeIndex = configs.EPISODE_START
        #self.currentEpisodeIndex = 0
        self.currentSimulationStep = 0
        self.vehicleList = []
        self.pedestrianList = []
        self.pedestrianIds = []
        self.collision = False

        print("connect to carla server ...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

    def addSensor(self, sensorType, location , rotation, vehicle):

        if(sensorType == 'Camera'):
            camera_init_trans = carla.Transform(location, rotation)
            # We create the camera through a blueprint that defines its properties
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            camera_bp.set_attribute('image_size_x', str(configs.IMAGE_X))
            camera_bp.set_attribute('image_size_y', str(configs.IMAGE_Y))
            camera_bp.set_attribute('sensor_tick', str(configs.SENSOR_TICK))
            camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to = vehicle)
            return camera
        
        elif(sensorType == 'Collision'):
            collisionSensor = self.world.spawn_actor(self.world.get_blueprint_library().find('sensor.other.collision'),
                                        carla.Transform(), attach_to= vehicle)
            return collisionSensor
        
        elif(sensorType == 'Lidar'):
            lidar_cam = None
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels',str(configs.LIDAR_CHANNELS))
            lidar_bp.set_attribute('points_per_second',str(configs.LIDAR_POINTS_PER_SEC))
            lidar_bp.set_attribute('rotation_frequency',str(configs.LIDAR_ROTATION_FREQ))
            lidar_bp.set_attribute('range',str(configs.LIDAR_RANGE))
            lidar_bp.set_attribute('upper_fov',str(configs.LIDAR_FOV_U))
            lidar_bp.set_attribute('lower_fov',str(configs.LIDAR_FOV_L))
            lidar_transform = carla.Transform(location, rotation)
            lidar_sen = self.world.spawn_actor(lidar_bp,lidar_transform,attach_to = vehicle)
            return lidar_sen
        
    def setup_episode_settings(self, town, traffic_denisty, pedestrian_denisty, weather, isTesting, startpoint, route_id = 0):
        
        # set episode settings
        succeed = True
        self.client.load_world(town)
        self.world.set_weather(weather)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = configs.SIM_TICK
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        used_spwan_points = []
        spawn_point = None

        self.trafficManager = self.client.get_trafficmanager(8000)
        self.trafficManager.set_synchronous_mode(True)

        self.trafficManager.global_percentage_speed_difference(45)

        # Get the blueprint library and filter for the vehicle and walkers blueprints
        vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle.*')
        vehicle_blueprints = [bp for bp in vehicle_blueprints if 'firetruck' not in bp.id.lower()]

        blueprintsWalkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        spawn_points = self.world.get_map().get_spawn_points()

        print("add ego vehicle")

        # Spawn ego vehicle
        ego_bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')

        if(isTesting):
            spawn_point = startpoint
            os.environ['PYTHONHASHSEED'] = str(42 + route_id)
            random.seed(42 + route_id)
            np.random.seed(42 + route_id)
            self.world.set_pedestrians_seed(42 +  route_id)
            self.trafficManager.set_random_device_seed(42 + route_id)
        else:
            spawn_point = random.choice(spawn_points)


        self.egoVehicle = self.world.try_spawn_actor(ego_bp, spawn_point)

        if self.egoVehicle is not None:
            self.vehicleList.append(self.egoVehicle)
            used_spwan_points.append(spawn_point)
        else:
            print("failed to spawn ego vehicle")
            succeed  = False

        print("add other vehicles")

        # Spawn vehicles randomly distributed throughout the map 
        # for each spawn point, we choose a random vehicle from the blueprint library
        for i in range(0, traffic_denisty):
            spawn_point = random.choice([item for item in spawn_points if item not in used_spwan_points])

            dis = np.sqrt((startpoint.location.x - spawn_point.location.x)**2 + (startpoint.location.y - spawn_point.location.y)**2)
            
            # avoid spawing any vehicles too close to the ego vehicle to avoid collision at the beginning of the simulation
            if dis < 5:
                continue

            actor = self.world.try_spawn_actor(random.choice(vehicle_blueprints), spawn_point)
            if actor is not None:
                self.vehicleList.append(actor)
                used_spwan_points.append(spawn_point)

        print(f"spawned {len(self.vehicleList) - 1} out of {traffic_denisty} vehicles")

        print("add cameras to ego vehicle")

        self.frontCamera = self.addSensor('Camera', carla.Location(2.0, 0.0, 1.4), carla.Rotation(-15.0, 0.0, 0.0), self.egoVehicle)
        self.leftCamera = self.addSensor('Camera', carla.Location(2.0, -2.0, 1.4), carla.Rotation(-15.0, -30.0, 0.0), self.egoVehicle)
        self.rightCamera = self.addSensor('Camera', carla.Location(2.0, 2.0, 1.4), carla.Rotation(-15.0, 30.0, 0.0), self.egoVehicle)

        #self.lidar = self.addSensor('Lidar', carla.Location(0, 0, 2.5), carla.Rotation(0, 0, 0), self.egoVehicle)
        
        if (self.frontCamera is None) or (self.frontCamera is None) or (self.rightCamera is None):
            print("failed to spawn sensors")
            succeed  = False
        else:
            self.frontCamera.listen(lambda data: self.cameraReadData(data, 'FrontCam'))
            self.leftCamera.listen(lambda data: self.cameraReadData(data, 'LeftCam'))
            self.rightCamera.listen(lambda data: self.cameraReadData(data, 'RightCam'))

        self.collisionSensor = self.addSensor('Collision', None, None, self.egoVehicle)

        if self.collisionSensor is None:
            print("failed to spawn collision sensor")
            succeed  = False
        else:
            self.collisionSensor.listen(lambda event: self.collisionHandler(event))

        
        print("add pedestrians")

        # Create spawn points for pedestrians    
        pedestrian_points = []
        for i in range(pedestrian_denisty):
            pedestrian_point = carla.Transform()
            pedestrian_point.location = self.world.get_random_location_from_navigation()

            if (pedestrian_point.location != None):
                pedestrian_points.append(pedestrian_point)

        batch = []

        for pedestrian_point in pedestrian_points:
            walker_bp = random.choice(blueprintsWalkers)
            batch.append(carla.command.SpawnActor(walker_bp, pedestrian_point))


        results = self.client.apply_batch_sync(batch, True)
        walkers_list = []

        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})

        # Spawn walker AI controllers for each walker
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        batch = []

        for i in range(len(walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

        # Apply the batch
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id

        self.pedestrianIds = []

        # Put altogether the walker and controller ids
        for i in range(len(walkers_list)):
            self.pedestrianIds.append(walkers_list[i]["con"])
            self.pedestrianIds.append(walkers_list[i]["id"])

        self.pedestrianList = self.world.get_actors(self.pedestrianIds)

        # Initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.pedestrianList), 2):
            # start walker
            self.pedestrianList[i].start()
            # set walk to random point
            self.pedestrianList[i].go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            self.pedestrianList[i].set_max_speed(1 + random.random())

        print(f"spawned {len(walkers_list)} pedestrians")

        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')

        #for traffic_light in traffic_lights:
            #traffic_light.set_red_time(5)
            #traffic_light.set_yellow_time(1)
            #traffic_light.set_green_time(5)

        # Make all vehicles move in autopilot mode
        ego_vehicle_id = self.egoVehicle.id  # Get the ID of the ego vehicle

        for vehicle in self.world.get_actors().filter('*vehicle*'):

            if vehicle.id == ego_vehicle_id and isTesting:
                continue
            else:
                vehicle.set_autopilot(True)

        #self.trafficManager.ignore_lights_percentage(self.egoVehicle, 100.0) # ego_vehicle ignores traffic lights
        self.trafficManager.distance_to_leading_vehicle(self.egoVehicle, 4.0)

        return succeed

    # Camera listener callback
    def cameraReadData(self, data, sensor):
        
        im = np.array(data.raw_data, dtype = np.dtype("uint8"))
        im = np.reshape(im, (configs.IMAGE_Y, configs.IMAGE_X, 4))
        im = im[:, :, :3]
        im = im[configs.IMAGE_CUT[0]: configs.IMAGE_CUT[1], :, :3]
        im = cv2.resize(im, dsize = (configs.IMAGE_RESIZE_X, configs.IMAGE_RESIZE_Y), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(configs.DATASET_PATH + '/episode_'+ str(self.currentEpisodeIndex)+'/' + str(sensor) + '/CameraRGB_'+str('{:08d}'.format(self.currentSimulationStep))+'.png', im)

    def collisionHandler(self, event):

        if(self.collision == False):
            print('warning: collion occured during episode = ' + str(self.currentEpisodeIndex) + ', skip episode')
            self.collision = True

    def get_distance_to_leading_vehicle(self, max_distance=20.0):

        ego_location = self.egoVehicle.get_location()
        ego_transform = self.egoVehicle.get_transform()
        ego_forward_vector = ego_transform.get_forward_vector()


        min_distance = float('inf')

        for vehicle in self.vehicleList:
            if vehicle.id == self.egoVehicle.id:
                continue


            vehicle_location = vehicle.get_location()
            vehicle_vector = vehicle_location - ego_location
            distance = vehicle_vector.length()

            if distance > max_distance:
                continue
            
            vehicle_vector_norm = vehicle_vector.make_unit_vector()
            ego_forward_vector_norm = ego_forward_vector.make_unit_vector()
            dot_product = ego_forward_vector_norm.x * vehicle_vector_norm.x + ego_forward_vector_norm.y * vehicle_vector_norm.y + ego_forward_vector_norm.z * vehicle_vector_norm.z

            # Check if the vehicle is in front of the ego vehicle using dot product
            vehicle_vector = vehicle_location - ego_location
            dot_product = ego_forward_vector.x * vehicle_vector.x + ego_forward_vector.y * vehicle_vector.y + ego_forward_vector.z * vehicle_vector.z

            angle = math.degrees(math.acos(max(min(dot_product, 1.0), -1.0))) * (180. / np.pi)  # from radian to degree

            if angle <= np.abs(10):  # consider the vehicles with slight angle to the ego vehcile to be ahead of the ego vehicle
                if distance < min_distance:
                    min_distance = distance

        return min_distance

    def collectData(self):

        lateral_noiser = Noiser(type = "lateral",
                                noise_min_interval= 0.25,
                                noise_max_interval= 0.50,
                                noise_min_intensity= 0.06,
                                noise_max_intensity= 0.12,
                                frequency= 1,
                                noise_probability= 0.4)
        
        longitudinal_noiser = Noiser(type = "longitudinal",
                                noise_min_interval= 1.0,
                                noise_max_interval= 2.0,
                                noise_min_intensity= 0.1,
                                noise_max_intensity= 0.2,
                                frequency= 1,
                                noise_probability= 0.25)
                               
        while(True):
     
            vehicle_transform = self.egoVehicle.get_transform()
            offset = carla.Location(x=-6.0, y=0.0, z=2.5)
            relative_location = vehicle_transform.transform(offset)
            rotation = carla.Rotation(
                            pitch=-15.0,
                            yaw=vehicle_transform.rotation.yaw,
                            roll=0.0)

            # Set the spectator transform
            spectator = self.world.get_spectator()
            spectator.set_transform(carla.Transform(relative_location, rotation))
                

            # Add noise to steering and limit speed
            control = self.egoVehicle.get_control()

            # speed from m/s to km/hr
            speed = 3.6 * np.sqrt((self.egoVehicle.get_velocity().x**2) + (self.egoVehicle.get_velocity().y**2))
            

            noisy_control = lateral_noiser.update_and_apply_noise(current_simulation_step= self.currentSimulationStep, control= control, speed= speed)

            # don't apply longitudinal noise if there is a very close vehicle ahead of the ego vehicle
            if self.get_distance_to_leading_vehicle(max_distance=20) >= 3.0:
                noisy_control = longitudinal_noiser.update_and_apply_noise(current_simulation_step= self.currentSimulationStep, control= control, speed= speed)


            self.egoVehicle.apply_control(noisy_control)

            self.world.tick()

            road_option = self.trafficManager.get_next_action(self.egoVehicle)[0]

            # Get navigationl command from traffic manager
            if(road_option == "LaneFollow"):
                navigation = 'F'
            elif(road_option == "Right"):
                navigation = 'R'
            elif(road_option == "Left"):
                navigation = 'L'
            elif(road_option == "Straight"):
                navigation = 'S'
            else:
                navigation = 'N'

            # save measurements to control csv file
            self.csv_writer.writerow([self.currentSimulationStep,
                                        self.egoVehicle.get_control().steer,
                                        self.egoVehicle.get_control().throttle,
                                        self.egoVehicle.get_control().brake,
                                        speed,
                                        self.egoVehicle.get_transform().rotation.yaw,
                                        self.egoVehicle.get_angular_velocity().z,
                                        self.egoVehicle.get_transform().location.x,
                                        self.egoVehicle.get_transform().location.y,
                                        navigation])
                    
            self.csv_file.flush()
            self.currentSimulationStep = self.currentSimulationStep + 1
            
            if(self.currentSimulationStep * configs.SIM_TICK >= configs.EPISODE_SIMULATION_DURATION_SEC) or self.collision:
                self.frontCamera.stop()
                self.leftCamera.stop()
                self.rightCamera.stop()
                self.collisionSensor.stop()
                return

    def makeEpisodeDir(self, episode):
            
        if not os.path.exists(os.path.join(configs.DATASET_PATH, 'episode_' + str(episode))):
            os.mkdir(os.path.join(configs.DATASET_PATH , 'episode_' + str(episode)))
            os.mkdir(os.path.join(configs.DATASET_PATH +'episode_' + str(episode), 'FrontCam'))
            os.mkdir(os.path.join(configs.DATASET_PATH +'episode_' + str(episode), 'LeftCam'))
            os.mkdir(os.path.join(configs.DATASET_PATH +'episode_' + str(episode), 'RightCam'))

            # ego vehicle control file
            self.csv_file = open('./dataset/episode_' + str(episode) + '/control.csv', 'w+')
            self.csv_writer = csv.writer(self.csv_file, delimiter=',')
            self.csv_writer.writerow(['step', 'steering_angle', 'throttle','brake', 'speed', 'yaw', 'yaw_rate', 'loc_x', 'loc_y', 'command'])
            self.csv_file.flush()

    def cleanup(self):

        print("episode cleanup")
    
        self.frontCamera.destroy()
        self.leftCamera.destroy()
        self.rightCamera.destroy()
        self.collisionSensor.destroy()

        for vehicle in self.vehicleList:
            vehicle.destroy()

        # stop pedestrians (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.pedestrianIds), 2):
            self.pedestrianList[i].stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.pedestrianIds])

        self.vehicleList = []
        self.pedestrianList = []
        self.pedestrianIds = []
        self.currentEpisodeIndex  = self.currentEpisodeIndex + 1
        self.currentSimulationStep = 0
        self.collision = False


def main():

    collector = dataCollector()

    for episode in range(configs.EPISODE_START, configs.EPISODE_COUNT):

        print("############################################################################")
        print('new episode  = ' + str(episode))

        collector.makeEpisodeDir(episode)

        traffic_denisty = np.random.randint(configs.TRAFFIC_DENISTY_MIN, configs.TRAFFIC_DENISTY_MAX)
        pedestrian_denisty = np.random.randint(configs.PEDESTRIAN_DESNISTY_MIN, configs.PEDESTRIAN_DESNISTY_MAX)
        weather = random.choice(configs.PREDEFINED_WEATHER_LIST)
        town = random.choice(configs.SIMULATION_TOWN_LIST)
    
        success = collector.setup_episode_settings(town, traffic_denisty, pedestrian_denisty, weather, False, 0)
        
        if success:
            print("start simulation")
            collector.collectData()
            collector.cleanup()
        else:
            print('skip episode  = ' + str(episode) + ' ...')
            collector.cleanup()
            continue

        time.sleep(10)



if __name__ == "__main__":
    main()
