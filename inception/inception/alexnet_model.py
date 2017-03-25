# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from inception.slim import slim
#slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
FLAGS = tf.app.flags.FLAGS

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Alex v2 model architecture.
  (Modified from inception_model.py)

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    None. As alexnet does not have Auxiliary Logits.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(alexnet_v2_arg_scope(weight_decay=0.0)):
      logits, endpoints = alexnet_v2(
          images,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  return logits, None # No auxiliary_logits


def loss(logits, labels, batch_size=None, aux_loss=False):
  """Adds all losses for the model.

  Note the final loss is not returned. Instead, the list of losses are collected
  by slim.losses. The losses are accumulated in tower_loss() and summed to
  calculate the total loss.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  num_classes = logits[0].get_shape()[-1].value
  dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)

  # Cross entropy loss for the main softmax prediction.
  slim.losses.cross_entropy_loss(logits[0],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=1.0)

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
  with tf.name_scope('summaries'):
    for act in endpoints.values():
      _activation_summary(act)


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc],
                      activation=tf.nn.relu,
                      bias=0.1,
                      weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d], padding='SAME'):
      with slim.arg_scope([slim.ops.max_pool], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
  """AlexNet version 2.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    # end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc, slim.ops.max_pool]):
      net = slim.ops.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.ops.max_pool(net, [3, 3], 2, scope='pool1')
      net = slim.ops.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.ops.max_pool(net, [3, 3], 2, scope='pool2')
      net = slim.ops.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.ops.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.ops.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.ops.max_pool(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.ops.conv2d],
                          stddev=0.005,
                          bias=0.1):
        net = slim.ops.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = slim.ops.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.ops.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.ops.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.ops.conv2d(net, num_classes, [1, 1],
                          activation=None,
                          batch_norm_params=None,
                          bias=0,
                          scope='fc8')

      # Convert end_points_collection into a end_point dict.
      end_points = {}
      #end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points['logits'] = net
      return net, end_points

default_image_size = 224

