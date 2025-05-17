import numpy as np
import configs
import trainingConfigs
import cv2
from keras.utils import Sequence
from imgaug import augmenters as iaa


# Use imgaug lib to create sequence of augmentors with diffierent probabilities 
def create_augmenter():

    st = lambda aug: iaa.Sometimes(0.4, aug)    # 40% of images to be augmented
    oc = lambda aug: iaa.Sometimes(0.3, aug)    # 30% of images to be augmented
    rl = lambda aug: iaa.Sometimes(0.1, aug)    # 10% of images to be augmented


    seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))),                                            # blur images with a sigma between 0 and 1.5
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),                             # randomly remove up to X% of the pixels
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)),

        # randomly remove up to X% of the pixels
        oc(iaa.Add((-40, 40), per_channel=0.5)),                                   # change brightness of images (by -X to Y of original value)
        st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),                            # change brightness of images (X-Y% of original value)
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),                # improve or worsen the contrast
    ], random_order=True)                                                          # order is shuffled each time with equal probability
    return seq
    

class DataGenerator(Sequence):

    'Generates data for Keras'
    def __init__(self, data, has_3cam, dataIds, batch_size = 32, shuffle = False, isAugmentation = False):

        'Initialization'
        self.dim = (88, 200, 3)
        self.batch_size = batch_size
        self.dataIds = dataIds
        self.data = data
        self.has_3cam = has_3cam
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmenter = create_augmenter()
        self.isAugmentation = isAugmentation
    

    def __len__(self):
        return int(np.floor(min(len(val) for val in self.dataIds.values()) / (self.batch_size / len(trainingConfigs.BRANCHES))))

    def __getitem__(self, index):

        'Generate one batch of data'
        batchIds = {key: None for key in trainingConfigs.BRANCHES}

        for branch in trainingConfigs.BRANCHES:
            batchIds[branch] = [self.dataIds[branch][k] for k in range(index * int(self.batch_size / len(trainingConfigs.BRANCHES)), (index + 1) * int(self.batch_size / len(trainingConfigs.BRANCHES)))]

        # Generate data-
        x, y = self.__data_generation(batchIds)
        return x, y
        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            for branch in trainingConfigs.BRANCHES:
                np.random.shuffle(self.dataIds[branch])


    def __data_generation(self, tempDataIds):
        'Generates data containing batch_size samples'

        # Initialization
        cam1 = np.empty((self.batch_size, *self.dim), dtype = np.dtype("uint8"))

        if self.has_3cam:
            cam2 = np.empty((self.batch_size, *self.dim), dtype = np.dtype("uint8"))
            cam3 = np.empty((self.batch_size, *self.dim), dtype = np.dtype("uint8"))

        speed = np.empty((self.batch_size), dtype = np.dtype("float32"))
        y = np.zeros((self.batch_size, 13), dtype = np.dtype("float32"))

        # Generate data
        for branch in trainingConfigs.BRANCHES:

            branch_idx = trainingConfigs.BRANCHES.index(branch) * int(self.batch_size / len(trainingConfigs.BRANCHES))
            
            for i in range(0, int(self.batch_size / len(trainingConfigs.BRANCHES))):
                
                idx = branch_idx + i

                cam1[idx,] = np.array(cv2.imread(configs.BRANCHES_DATASET_PATH + branch + '/FrontCam/' + str(tempDataIds[branch][i]) + '.png'))
                if self.has_3cam:
                    cam2[idx,] = np.array(cv2.imread(configs.BRANCHES_DATASET_PATH + branch + '/LeftCam/' + str(tempDataIds[branch][i]) + '.png'))
                    cam3[idx,] = np.array(cv2.imread(configs.BRANCHES_DATASET_PATH + branch + '/RightCam/' + str(tempDataIds[branch][i]) + '.png'))

                output_sample = self.data[branch].loc[tempDataIds[branch][i]]
                
                y[idx, 3 * trainingConfigs.BRANCHES.index(branch)] = output_sample['steering_angle']
                y[idx, 3 * trainingConfigs.BRANCHES.index(branch) + 1] = output_sample['throttle'] 
                y[idx, 3 * trainingConfigs.BRANCHES.index(branch) + 2] = output_sample['brake']
                y[idx, -1] = output_sample['speed'] / trainingConfigs.MAX_FORWARD_SPEED
                speed[idx,] = output_sample['speed'] / trainingConfigs.MAX_FORWARD_SPEED

            if self.isAugmentation:
                cam1 = self.augmenter.augment_images(cam1)
                if self.has_3cam:
                    cam2 = self.augmenter.augment_images(cam2)
                    cam3 = self.augmenter.augment_images(cam3)

        if self.has_3cam:
            return [cam1 / 255., cam2 / 255., cam3 / 255., speed], y
        else:
            return [cam1 / 255., speed], y