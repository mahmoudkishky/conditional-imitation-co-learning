from dataCollector import dataCollector
import configs
import numpy as np
import random
import json
import cilModel
import cv2
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model
import os
import csv
import argparse
import trainingConfigs
from utils import decode_sine_wave
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
import carla
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Don't use the GPU @ carla client side while testing

weather_conditions = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
}

timeout = {
    "straight": 120.0,
    "one_curve": 120.0,
    "navigation": 10 * 60.0,
}

dynamic_navigation_traffic = {
    "Town01": {"vehicles": 50, "pedetrians": 100},
    "Town02": {"vehicles": 30, "pedetrians": 60},
}

class testModel:

    def modelTestReadCamData(self, data, sensor):
        im = np.array(data.raw_data, dtype = np.dtype("uint8"))
        im = np.reshape(im, (configs.IMAGE_Y, configs.IMAGE_X, 4))
        im = im[:, :, :3]
        im = im[configs.IMAGE_CUT[0]: configs.IMAGE_CUT[1], :, :3]
        im = cv2.resize(im, dsize = (configs.IMAGE_RESIZE_X, configs.IMAGE_RESIZE_Y), interpolation = cv2.INTER_CUBIC)
        
        if(sensor == 'FrontCam'):
            self.front_img = im / 255.
        elif(sensor == 'LeftCam'):
            self.left_img = im / 255.
        elif(sensor == 'RightCam'):
            self.right_img = im / 255.


    def modelTestCollisionHandler(self, event):

        if(self.collector.collision == False):
            print('WARN: collion occured during episode')
            self.collector.collision = True

    def __init__(self):
        self.collector = dataCollector()
        self.collector.cameraReadData = self.modelTestReadCamData
        self.collector.collisionHandler = self.modelTestCollisionHandler


def cil_step(roadOption):
        
    global test_environment
    global model

    if(roadOption == RoadOption.LEFT):
        branch = 0
    elif(roadOption == RoadOption.RIGHT):
        branch = 1
    elif(roadOption == RoadOption.STRAIGHT):
        branch = 2
    else:
        branch = 3

    vehicle_speed = 3.6 * np.sqrt(test_environment.collector.egoVehicle.get_velocity().x**2 + test_environment.collector.egoVehicle.get_velocity().y**2)
    predicted_actions = model.predict([np.reshape(test_environment.front_img, (1,88,200,3)), np.reshape(test_environment.left_img, (1,88,200,3)), np.reshape(test_environment.right_img, (1,88,200,3)), np.array([vehicle_speed]).reshape(1, 1)/trainingConfigs.MAX_FORWARD_SPEED], verbose = 0)
    steer, throttle, brake = predicted_actions[0][3 * branch:  3 * (branch + 1) ].tolist()
    predicted_speed = trainingConfigs.MAX_FORWARD_SPEED * predicted_actions[0][-1].tolist() 

    if trainingConfigs.ASINE_ENCODING and (roadOption == RoadOption.LEFT or roadOption == RoadOption.RIGHT):
        steer = np.sin((np.pi / 2) * steer)

    # don't go too fast
    if vehicle_speed > 25.0:
        throttle = throttle / 2.0
        brake = 0.8

    # avoid cil inertia problem 
    if brake < 0.1 or throttle > brake:
        brake = 0.0
    
    if vehicle_speed < 0.5 and predicted_speed > 1.5:
        throttle = throttle + 1 * ((10.0 - vehicle_speed) / trainingConfigs.MAX_FORWARD_SPEED)
        brake = 0.0
    
    return steer, throttle, brake
    

def parse_waypoints_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    routes = []
    for route in root.findall("route"):
        waypoints = []
        route_id = route.get("id")
        for waypoint in route.findall("waypoint"):
            x = float(waypoint.get("x"))
            y = float(waypoint.get("y"))
            z = float(waypoint.get("z"))
            yaw = float(waypoint.get("yaw"))
            pitch = float(waypoint.get("pitch", 0.0))
            roll = float(waypoint.get("roll", 0.0))
            location = carla.Location(x=x, y=y, z=z)
            rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
            waypoints.append(carla.Transform(location, rotation))

        routes.append((route_id, waypoints[0], waypoints[-1]))
    
    return routes

def main():

    
    global test_environment
    global model

    test_environment = testModel()

    csv_file = open('./outputs/results_{}_{}.csv'.format(args.town, args.task), 'w+')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['Route_id', 'Town', 'Weather', 'Distance', 'ReachDesitination'])

    csv_file.flush()

    xml_config = "./corl2017/{}_{}.xml".format(args.town, args.task)
    routes = parse_waypoints_from_xml(xml_config)



    for weather in args.weather:

        test_environment.collector.currentEpisodeIndex = 0

        for route in routes:

            route_id  = int(route[0])
            path = (route[1], route[2])

            if args.dynamic:
                traffic_denisty = dynamic_navigation_traffic[args.town]["vehicles"]
                pedestrian_denisty = dynamic_navigation_traffic[args.town]["pedetrians"]
            else:
                traffic_denisty = 0
                pedestrian_denisty = 0

            success = test_environment.collector.setup_episode_settings(args.town, traffic_denisty, pedestrian_denisty, weather_conditions[weather], True, path[0], route_id)

            with open('./outputs/cil_model.json', 'r') as jfile:
                model = model_from_json(json.load(jfile))

            model.load_weights("./outputs/cil_model.keras")
            

            if(success):

                agent = BasicAgent(test_environment.collector.egoVehicle)
                distance  = 0
                spectator = test_environment.collector.world.get_spectator()
                agent.set_destination(path[1].location)

                while True:

                    test_environment.collector.world.tick()

                    vehicle_transform = test_environment.collector.egoVehicle.get_transform()
                    offset = carla.Location(x=-6.0, y=0.0, z=4.0)
                    relative_location = vehicle_transform.transform(offset)
                    rotation = carla.Rotation(
                        pitch=-15.0,
                        yaw=vehicle_transform.rotation.yaw,
                        roll=0.0
                    )

                    # Set the spectator transform
                    spectator.set_transform(carla.Transform(relative_location, rotation))

                    test_environment.collector.currentSimulationStep += 1

                    
                    if agent.done() or test_environment.collector.collision or test_environment.collector.currentSimulationStep * configs.SIM_TICK >= timeout[args.task]:
                    
                        csv_writer.writerow([route_id, args.town, weather, distance, str(agent.done())])
                        csv_file.flush()
                        test_environment.collector.cleanup()
                        print("##########################################################")

                        break
                    
                    control = agent.run_step()
                    direction = agent.get_local_planner().get_incoming_waypoint_and_direction(steps = 0)[1]

                    steer, throttle, brake = cil_step(direction)
                    control.steer =  steer
                    control.throttle = throttle
                    control.brake =  brake

                    current_speed = np.sqrt(test_environment.collector.egoVehicle.get_velocity().x**2 + test_environment.collector.egoVehicle.get_velocity().y**2)
                    distance += current_speed * configs.SIM_TICK 
                    test_environment.collector.egoVehicle.apply_control(control)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='corl2017 benchmark')
    parser.add_argument('--town', dest='town', choices = ['Town01', 'Town02'], help='testing town')
    parser.add_argument('--task', dest='task', choices = ['straight', 'one_curve', 'navigation'])
    parser.add_argument('--dynamic', dest='dynamic', action = 'store_true', default= False)
    parser.add_argument('--weather', dest='weather', nargs= '+', choices = ['ClearNoon', 'WetNoon', 'WetCloudyNoon', 'HardRainNoon', 'ClearSunset', 'SoftRainSunset'])
    args = parser.parse_args()
    main()



