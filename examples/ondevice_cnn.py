# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI wrapper for tflite_transfer_converter.

Converts a TF model to a TFLite transfer learning model.
"""

import os

import numpy as np
import tensorflow as tf

INPUT_CHANNELS = 6
SEG_SIZE = 300
NUM_FEATURES = 1 * 1 * 128
MAX_CLASSES = 20 # new activity가 들어올때마다 모델 구조를 바꾸는게 시간이 더 걸림 그냥 적절한 맥시멈 헤드 수를 정해놓고 마스킹
NUM_CLASSES = 18
NUM_NEW_CLASSES = 2


class OnDeviceLearningModel(tf.Module):
  """On-Device CL model class."""

  def __init__(self, backbone, initial_out, learning_rate=0.001):
    """
    :param backbone: non-trainable feature extractor
    :param initial_out: the number of initial classes
    """

    self.num_features = NUM_FEATURES
    self.max_heads = MAX_CLASSES
    self.num_classes = initial_out
    self.mask_value = -1000

    # masking
    self.active_units = tf.Variable(tf.zeros(self.max_heads, dtype=tf.bool))
    self.active_units[:self.num_classes].assign(True)

    # trainable weights and bias for softmax
    self.ws = tf.Variable(
        tf.zeros((self.num_features, self.max_heads)),
        name='ws',
        trainable=True)
    self.bs = tf.Variable(
        tf.zeros((1, self.max_heads)), name='bs', trainable=True)

    # base model
    self.base = backbone
    # loss function and optimizer
    self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  @tf.function(input_signature=[])
  def adapt(self):
    self.num_classes += NUM_NEW_CLASSES
    self.active_units[:self.num_classes].assign(True)
    return True

  @tf.function(input_signature=[
      tf.TensorSpec([None, SEG_SIZE, INPUT_CHANNELS], tf.float32),
  ])
  def load(self, feature):
    """Generates and loads bottleneck features from the given image batch.

    Args:
      feature: A tensor of image feature batch to generate the bottleneck from.

    Returns:
      Map of the bottleneck.
    """
    bottleneck = tf.reshape(
        self.base(feature, training=False), (-1, self.num_features))
    return {'bottleneck': bottleneck}


  @tf.function(input_signature=[
      tf.TensorSpec([None, NUM_FEATURES], tf.float32),
      tf.TensorSpec([None, MAX_CLASSES], tf.float32),
  ])
  def train(self, bottleneck, label):
    """Runs one training step with the given bottleneck features and labels.

    Args:
      bottleneck: A tensor of bottleneck features generated from the base model.
      label: A tensor of class labels for the given batch.

    Returns:
      Map of the training loss.
    """
    with tf.GradientTape() as tape:
      logits = tf.matmul(bottleneck, self.ws) + self.bs
      #mask = tf.logical_not(self.active_units)
      #logits = tf.mask_fill(mask=mask, value=self.mask_value)
      logits = tf.where(self.active_units, logits, self.mask_value)
      prediction = tf.nn.softmax(logits)
      loss = self.loss_fn(prediction, label)
    gradients = tape.gradient(loss, [self.ws, self.bs])
    self.optimizer.apply_gradients(zip(gradients, [self.ws, self.bs]))
    result = {'loss': loss}
    for grad in gradients:
      result[grad.name] = grad
    return result

  @tf.function(input_signature=[
      tf.TensorSpec([None, SEG_SIZE, INPUT_CHANNELS], tf.float32)
  ])
  def infer(self, feature):
    """Invokes an inference on the given feature.

    Args:
      feature: A tensor of image feature batch to invoke an inference on.

    Returns:
      Map of the softmax output.
    """
    bottleneck = tf.reshape(
        self.base(feature, training=False), (-1, self.num_features))
    logits = tf.matmul(bottleneck, self.ws) + self.bs
    logits = tf.where(self.active_units, logits, self.mask_value)
    return {'output': tf.nn.softmax(logits)}

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def save(self, checkpoint_path):
    """Saves the trainable weights to the given checkpoint file.

    Args:
      checkpoint_path: A file path to save the model.

    Returns:
      Map of the checkpoint file path.
    """
    tensor_names = [self.ws.name, self.bs.name]
    tensors_to_save = [self.ws.read_value(), self.bs.read_value()]
    tf.raw_ops.Save(
        filename=checkpoint_path,
        tensor_names=tensor_names,
        data=tensors_to_save,
        name='save')
    return {'checkpoint_path': checkpoint_path}

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def restore(self, checkpoint_path):
    """Restores the serialized trainable weights from the given checkpoint file.

    Args:
      checkpoint_path: A path to a saved checkpoint file.

    Returns:
      Map of restored weight and bias.
    """
    restored_tensors = {}
    restored = tf.raw_ops.Restore(
        file_pattern=checkpoint_path,
        tensor_name=self.ws.name,
        dt=np.float32,
        name='restore')
    self.ws.assign(restored)
    restored_tensors['ws'] = restored
    restored = tf.raw_ops.Restore(
        file_pattern=checkpoint_path,
        tensor_name=self.bs.name,
        dt=np.float32,
        name='restore')
    self.bs.assign(restored)
    restored_tensors['bs'] = restored
    return restored_tensors

  @tf.function(input_signature=[])
  def initialize_weights(self):
    """Initializes the weights and bias of the head model.

    Returns:
      Map of initialized weight and bias.
    """
    self.ws.assign(tf.random.uniform((self.num_features, self.max_heads)))
    self.bs.assign(tf.random.uniform((1, self.max_heads)))
    return {'ws': self.ws, 'bs': self.bs}


def convert_and_save(backbone, init_out, saved_model_dir='saved_model'):
  """Converts and saves the TFLite Transfer Learning model.

  Args:
    saved_model_dir: A directory path to save a converted model.
  """
  model = OnDeviceLearningModel(backbone, init_out)

  tf.saved_model.save(
      model,
      saved_model_dir,
      signatures={
          'load': model.load.get_concrete_function(),
          'train': model.train.get_concrete_function(),
          'infer': model.infer.get_concrete_function(),
          'save': model.save.get_concrete_function(),
          'restore': model.restore.get_concrete_function(),
          'initialize': model.initialize_weights.get_concrete_function(),
          'adapt': model.adapt.get_concrete_function(),
      })

  # Convert the model
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  converter.experimental_enable_resource_variables = True
  tflite_model = converter.convert()

  model_file_path = os.path.join('model.tflite')
  with open(model_file_path, 'wb') as model_file:
    model_file.write(tflite_model)


if __name__ == '__main__':
  loadModel = tf.keras.models.load_model("pretrained")
#   model = OnDeviceLearningModel(loadModel.backbone, )
  
#   print(type(model.ws))
#   model.adapt()
#   print(type(model.ws))
#   print(model.ws.read_value().shape)
  convert_and_save(loadModel.backbone, 10)