import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import shutil
import configs
import trainingConfigs

def main():

    cameras = ['FrontCam', 'LeftCam', 'RightCam']
    
    for branch in trainingConfigs.BRANCHES:
        
        print('branch: ', branch)
        id  = 0
        csv_file = open(configs.BRANCHES_DATASET_PATH + branch + '/control.csv', 'w+')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['step', 'steering_angle', 'throttle','brake', 'speed', 'yaw', 'yaw_rate', 'loc_x', 'loc_y', 'command'])
        csv_file.flush()
        
        for episode in range(configs.EPISODE_COUNT):
            if os.path.exists(configs.DATASET_PATH + 'episode_' + str(episode)):
            
                print('episode: ',episode)

                df = pd.read_csv(configs.DATASET_PATH + 'episode_' + str(episode) + "/control.csv")
                df = df[::2] # Drop all odd entries in csv file -> no corresponding cameras captures
                branch_df = df.loc[df['command'] == branch]

                for step in list(branch_df["step"]):

                    valid = True
                    
                    for camera in cameras:
                        im_path = configs.DATASET_PATH + 'episode_' + str(episode) +'/' + camera + '/CameraRGB_'+  str('{:08d}'.format(step)) + '.png'
                        
                        if os.path.isfile(im_path):
                            im_new_path = configs.BRANCHES_DATASET_PATH + branch + '/' + camera +'/'+ str(id) + '.png'
                            shutil.move(im_path, im_new_path)
                            #shutil.copyfile(im_path, im_new_path)
                        else:
                            valid = False
                        
                    if valid:    
                        csv_writer.writerow([id,
                                        branch_df.loc[step,'steering_angle'],
                                        branch_df.loc[step,'throttle'],
                                        branch_df.loc[step,'brake'],
                                        branch_df.loc[step,'speed'],
                                        branch_df.loc[step,'yaw'],
                                        branch_df.loc[step,'yaw_rate'],
                                        branch_df.loc[step,'loc_x'],
                                        branch_df.loc[step,'loc_y'],
                                        branch_df.loc[step,'command']])

                        csv_file.flush()
                        id  = id + 1


if __name__ == "__main__":
    main()