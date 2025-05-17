import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout, LeakyReLU, Conv2D, ReLU, Activation
from tensorflow.keras.activations import tanh, softmax, sigmoid
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow import one_hot
from tensorflow.keras.layers import Concatenate, Add
import numpy as np
import scipy.stats as stats
import trainingConfigs
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Lambda

def cil_loss(y_true, y_pred):
  
  batch_nav_cmds = (
      [0] * (trainingConfigs.BATCH_SIZE // len(trainingConfigs.BRANCHES)) +       # left
      [1] * (trainingConfigs.BATCH_SIZE // len(trainingConfigs.BRANCHES)) +       # right
      [2] * (trainingConfigs.BATCH_SIZE // len(trainingConfigs.BRANCHES)) +       # straight
      [3] * (trainingConfigs.BATCH_SIZE // len(trainingConfigs.BRANCHES))         # follow lane
  )

  batch_nav_cmds = tf.convert_to_tensor(batch_nav_cmds, dtype=tf.int32)
  nav_cmds_mask = tf.one_hot(batch_nav_cmds, depth=4)  # Shape: (batch_size, 4)

  branch_weights = tf.constant([trainingConfigs.STEER_LOSS_WEIGHT,
                                trainingConfigs.THROTTLE_LOSS_WEIGHT,
                                trainingConfigs.BRAKE_LOSS_WEIGHT], dtype=tf.float32)
  
  branch_weights = tf.reshape(branch_weights, [1, 1, -1])

  nav_cmds_mask_expanded = tf.expand_dims(nav_cmds_mask, axis=-1)  # Shape: (batch_size, 4, 1)
  branch_mask = nav_cmds_mask_expanded * branch_weights  # Shape: (batch_size, 4, 3)

  branch_mask = tf.reshape(branch_mask, [(trainingConfigs.BATCH_SIZE), len(trainingConfigs.BRANCHES) * 3]) # Shape: (batch_size, 12)
  # Add speed weight
  speed_mask = tf.ones(((trainingConfigs.BATCH_SIZE), 1), dtype=tf.float32) * trainingConfigs.SPEED_LOSS_WEIGHT  # Shape: (batch_size, 1)
  weighted_mask = tf.concat([branch_mask, speed_mask], axis=-1)  # Shape: (batch_size, 13)

  return tf.reduce_mean(weighted_mask * tf.abs(y_true - y_pred))
  #return tf.reduce_mean(weighted_mask * tf.square(y_true - y_pred))


def camera_stream(img):
    
  cnn = Conv2D(32, (5, 5), (2,2), input_shape=(88, 200, 3), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT))(img)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = Conv2D(32, (3, 3), (1,1), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = Conv2D(64, (3, 3), (2,2), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = Conv2D(64, (3, 3), (1,1), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = Conv2D(128, (3, 3), (2,2), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = Conv2D(128, (3, 3), (1,1), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)


  cnn = Conv2D(256, (3, 3), (1,1), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT) )(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = Conv2D(256, (3, 3), (1,1), kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.CONV_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = BatchNormalization()(cnn)
    
  cnn = Flatten()(cnn)

  cnn = Dense(512, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = Dropout(0.3)(cnn)

  cnn = Dense(512, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(cnn)
  cnn = LeakyReLU()(cnn)
  cnn = Dropout(0.3)(cnn)
  

  return cnn


def measurement_stream(meas):

  meas = Dense(128, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(meas)
  meas = LeakyReLU()(meas)
  meas = Dropout(0.5)(meas)

  meas = Dense(128, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(meas)
  meas = LeakyReLU()(meas)
  meas = Dropout(0.5)(meas)

  return meas


def ColearningGatedUnit(input, units):


  nn = Dense(trainingConfigs.GTU_SIZE, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(input)
  nn = LeakyReLU()(nn)
  nn = Dropout(0.5)(nn)

  nn = Dense(trainingConfigs.GTU_SIZE, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(nn)
  nn = LeakyReLU()(nn)
  nn = Dropout(0.5)(nn)

  nn = Dense(units, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(nn)
  nn = Activation(tanh)(nn)
  

  gate = Dense(trainingConfigs.GTU_SIZE, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(input)
  gate = LeakyReLU()(gate)
  gate = Dropout(0.5)(gate)

  gate = Dense(trainingConfigs.GTU_SIZE, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(gate)
  gate = LeakyReLU()(gate)
  gate = Dropout(0.5)(gate)

  gate = Dense(units, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(gate)
  gate = Activation(sigmoid)(gate)

  return gate * nn


def single_cam_model(cic = False):


  branches = []
  gtus = []
  br = []

  front_input_img = Input(shape=(88, 200, 3))
  meas_speed = Input(shape=(1))

  # images from the front-facing camera
  cam = camera_stream(front_input_img)

  # measurements
  meas  = measurement_stream(meas_speed)

  # sensor fusion
  fus = tf.concat([cam, meas], 1)
  fus = Dense(512, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(fus)
  fus = LeakyReLU()(fus)
  fus = Dropout(0.5)(fus)


  # auxiliary output (predicted speed only from the captured images)
  aux = Dense(256, kernel_initializer = GlorotNormal(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(cam)
  aux = LeakyReLU()(aux)
  aux = Dropout(0.5)(aux)

  aux = Dense(256, kernel_initializer = GlorotNormal(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(aux)
  aux = LeakyReLU()(aux)
  aux = Dropout(0.5)(aux)

  auxiliary_output = Dense(1)(aux)
  

  # predited actions (steering, throttle, brake) for branches (go left, go right, go straight, follow lane)
  for i in range(len(trainingConfigs.BRANCHES)):

    branch = Dense(256, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(fus)
    branch = LeakyReLU()(branch)
    branch = Dropout(0.5)(branch)

    branch = Dense(256, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(branch)
    branch = LeakyReLU()(branch)
    branch = Dropout(0.5)(branch)
    
    if cic:

      #g = ColearningGatedUnit(branch, len(trainingConfigs.BRANCHES) - 1)
      g = ColearningGatedUnit(branch, 2 * (len(trainingConfigs.BRANCHES) - 1))  #for steering and brake

      """
      gtu = tf.concat([
          g[:, 0:1], # gtu for first branch steering
          tf.ones((tf.shape(branch)[0], 2)),  # pad throttle & brake with ones
          g[:, 1:2],
          tf.ones((tf.shape(branch)[0], 2)),
          g[:, 2:3],
          tf.ones((tf.shape(branch)[0], 2)),
      ], axis=1) 
      """
      gtu = tf.concat([
          g[:, 0:1], # gtu for first branch steering
          tf.ones((tf.shape(branch)[0], 1)),  # pad throttle
          g[:, 1:2], # gtu for first branch brake

          g[:, 2:3],
          tf.ones((tf.shape(branch)[0], 1)),
          g[:, 3:4],

          g[:, 4:5],
          tf.ones((tf.shape(branch)[0], 1)),
          g[:, 5:6],
      ], axis=1) 

      gtus.append(gtu)
      br.append(Dense(3, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(branch))

    else:
      branches.append(Dense(3, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(branch))
  
  
  if cic:
          
      branches.append(Add()([br[0], tf.reduce_sum(Concatenate()([br[1], br[2], br[3]]) * gtus[0], axis = 1)]))
      branches.append(Add()([br[1], tf.reduce_sum(Concatenate()([br[0], br[2], br[3]]) * gtus[1], axis = 1)]))
      branches.append(Add()([br[2], tf.reduce_sum(Concatenate()([br[0], br[1], br[3]]) * gtus[2], axis = 1)]))
      branches.append(Add()([br[3], tf.reduce_sum(Concatenate()([br[0], br[1], br[2]]) * gtus[3], axis = 1)]))
  


  combined_outputs = Concatenate()([branches[0], branches[1], branches[2], branches[3], auxiliary_output])
  model = Model(inputs=[front_input_img, meas_speed], outputs= [combined_outputs])
  opt  = Adam(learning_rate = trainingConfigs.LEARNING_RATE, beta_1 = trainingConfigs.BETA1, beta_2 = trainingConfigs.BETA2)
  
  model.compile(optimizer = opt, loss = cil_loss)  
  
  return model




def three_cam_model(cic = False):


  branches = []
  gtus = []
  br = []

  front_input_img = Input(shape=(88, 200, 3))
  left_input_img = Input(shape=(88, 200, 3))
  right_input_img = Input(shape=(88, 200, 3))

  meas_speed = Input(shape=(1))

  # images from left, front and right cameras
  cam1 = camera_stream(front_input_img)
  cam2 = camera_stream(left_input_img)
  cam3 = camera_stream(right_input_img)
  cam = tf.concat([cam1, cam2, cam3], 1)

  # measurements
  meas  = measurement_stream(meas_speed)

  # sensor fusion
  fus = tf.concat([cam, meas], 1)
  fus = Dense(512, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(fus)
  fus = LeakyReLU()(fus)
  fus = Dropout(0.5)(fus)


  # auxiliary output (predicted speed only from the captured images)
  aux = Dense(256, kernel_initializer = GlorotNormal(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(cam)
  aux = LeakyReLU()(aux)
  aux = Dropout(0.5)(aux)

  aux = Dense(256, kernel_initializer = GlorotNormal(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(aux)
  aux = LeakyReLU()(aux)
  aux = Dropout(0.5)(aux)

  auxiliary_output = Dense(1)(aux)
  

  # predited actions (steering, throttle, brake) for branches (go left, go right, go straight, follow lane)
  for i in range(len(trainingConfigs.BRANCHES)):

    branch = Dense(256, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(fus)
    branch = LeakyReLU()(branch)
    branch = Dropout(0.5)(branch)

    branch = Dense(256, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(branch)
    branch = LeakyReLU()(branch)
    branch = Dropout(0.5)(branch)
    
    if cic:

      #g = ColearningGatedUnit(branch, len(trainingConfigs.BRANCHES) - 1)
      g = ColearningGatedUnit(branch, 2 * (len(trainingConfigs.BRANCHES) - 1))  #for steering and brake

      """
      gtu = tf.concat([
          g[:, 0:1], # gtu for first branch steering
          tf.ones((tf.shape(branch)[0], 2)),  # pad throttle & brake with ones
          g[:, 1:2],
          tf.ones((tf.shape(branch)[0], 2)),
          g[:, 2:3],
          tf.ones((tf.shape(branch)[0], 2)),
      ], axis=1) 
      """
      gtu = tf.concat([
          g[:, 0:1], # gtu for first branch steering
          tf.ones((tf.shape(branch)[0], 1)),  # pad throttle
          g[:, 1:2], # gtu for first branch brake

          g[:, 2:3],
          tf.ones((tf.shape(branch)[0], 1)),
          g[:, 3:4],

          g[:, 4:5],
          tf.ones((tf.shape(branch)[0], 1)),
          g[:, 5:6],
      ], axis=1) 

      gtus.append(gtu)
      br.append(Dense(3, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(branch))

    else:
      branches.append(Dense(3, kernel_initializer = GlorotUniform(), bias_initializer = Constant(trainingConfigs.FC_BIAS_INIT))(branch))
  
  
  if cic:
          
      branches.append(Add()([br[0], tf.reduce_sum(Concatenate()([br[1], br[2], br[3]]) * gtus[0], axis = 1)]))
      branches.append(Add()([br[1], tf.reduce_sum(Concatenate()([br[0], br[2], br[3]]) * gtus[1], axis = 1)]))
      branches.append(Add()([br[2], tf.reduce_sum(Concatenate()([br[0], br[1], br[3]]) * gtus[2], axis = 1)]))
      branches.append(Add()([br[3], tf.reduce_sum(Concatenate()([br[0], br[1], br[2]]) * gtus[3], axis = 1)]))
  


  combined_outputs = Concatenate()([branches[0], branches[1], branches[2], branches[3], auxiliary_output])
  model = Model(inputs=[front_input_img, left_input_img, right_input_img, meas_speed], outputs= [combined_outputs])
  opt  = Adam(learning_rate = trainingConfigs.LEARNING_RATE, beta_1 = trainingConfigs.BETA1, beta_2 = trainingConfigs.BETA2)
  
  model.compile(optimizer = opt, loss = cil_loss)  
  
  return model