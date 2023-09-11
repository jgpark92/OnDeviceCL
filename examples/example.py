import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD

print("TensorFlow version:", tf.__version__)

data_path = "OnDeviceCL/Data/KU-HAR/deploy"
train_X = np.load(data_path+'/train_x.npy')
train_Y = np.load(data_path+'/train_y.npy')
test_X = np.load(data_path+'/test_x.npy')
test_Y = np.load(data_path+'/test_y.npy')

train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

input_nc = 6
segment_size = 300
class_num = 18

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.backbone = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, strides=1, padding='SAME', activation='relu', input_shape=(segment_size, input_nc)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, 3, strides=1, padding='SAME', activation='relu'),
        tf.keras.layers.Conv1D(128, 3, strides=1, padding='SAME', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last'),
    ])

    self.fc = tf.keras.layers.Dense(class_num, name='dense_1')

  def call(self, x, training=True):
    features = self.backbone(x)
    return self.fc(features)
  
NUM_EPOCHS = 300
epochs = np.arange(1, NUM_EPOCHS + 1, 1)
losses = np.zeros([NUM_EPOCHS])
m = Model()

train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
train_ds = train_ds.batch(BATCH_SIZE)

opt = SGD(lr=0.01)
m.compile(
  optimizer = opt,
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy']
)

def decay(epoch):
    if epoch < 150:
      return 0.1
    if epoch >= 150 and epoch < 225:
      return 0.01
    if epoch >= 225:
      return 0.001

m.fit(train_dataset, epochs=100, validation_data=test_dataset,
        verbose=2, callbacks=[tf.keras.callbacks.LearningRateScheduler(
            decay)])

m.save("pretrained")