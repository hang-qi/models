# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Builds the all-cnn C network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 16
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.05      # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


# From slim.variables
# Collection containing all the variables created using slim.variables
MODEL_VARIABLES = '_model_variables_'

# Collection containing the slim.variables that are created with restore=True.
VARIABLES_TO_RESTORE = '_variables_to_restore_'

LOSSES_COLLECTION = '_losses'

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer,
                     trainable=True, collections=None, restore=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    collections: A list of collection names to which the Variable will be added.
      Note that the variable is always also added to the tf.GraphKeys.GLOBAL_VARIABLES
      and MODEL_VARIABLES collections.
    restore: whether the variable should be added to the
      VARIABLES_TO_RESTORE collection.
  Returns:
    Variable Tensor
  """

  # The following snippets and
  # trainable=True, collections=None, restore=True is copied from slim.variables
  collections = list(collections or [])

  # Make sure variables are added to tf.GraphKeys.GLOBAL_VARIABLES and MODEL_VARIABLES
  collections += [tf.GraphKeys.GLOBAL_VARIABLES, MODEL_VARIABLES]
  # Add to VARIABLES_TO_RESTORE if necessary
  if restore:
    collections.append(VARIABLES_TO_RESTORE)
  # Remove duplicates
  collections = set(collections)
  # Slim variable done.

  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype,
                          trainable=trainable, collections=collections)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection(LOSSES_COLLECTION, weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size,
                                                  image_size=IMAGE_SIZE)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size,
                                        image_size=IMAGE_SIZE)
  return images, labels


def inference(inputs, num_classes=NUM_CLASSES, for_training=False,
                     restore_logits=True, scope=None):
  """Build the allcnn model.

  Dropout(keep=.8),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((1, 1, 192), **conv),
          Conv((1, 1, 16), **conv),
          Pooling(8, op="avg"),
          Activation(Softmax())
  """
  with tf.name_scope(scope, 'allcnn', [inputs]) as sc:
    with slim.arg_scope([slim.ops.conv2d],
        weight_decay=0.001, padding='SAME', stride=1, stddev=0.05):
      net = slim.ops.dropout(inputs, 0.8, is_training=for_training, scope='dp1')
      net = slim.ops.conv2d(net, 96, [3, 3], scope='conv1')
      net = slim.ops.conv2d(net, 96, [3, 3], scope='conv2')
      net = slim.ops.conv2d(net, 96, [3, 3], stride=2, scope='conv3')
      net = slim.ops.dropout(net, 0.5, is_training=for_training, scope='dp2')
      net = slim.ops.conv2d(net, 192, [3, 3], scope='conv4')
      net = slim.ops.conv2d(net, 192, [3, 3], scope='conv5')
      net = slim.ops.conv2d(net, 192, [3, 3], stride=2, scope='conv6')
      net = slim.ops.dropout(net, 0.5, is_training=for_training, scope='dp3')
      net = slim.ops.conv2d(net, 192, [3, 3], scope='conv7')
      net = slim.ops.conv2d(net, 192, [1, 1], padding='VALID', scope='conv8')
      net = slim.ops.conv2d(net, 16, [1, 1], padding='VALID', scope='conv9')
      net = slim.ops.avg_pool(net, 8, padding='VALID', scope='avg_pool')

      # linear layer(WX + b),
      # We don't apply softmax here because
      # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
      # and performs the softmax internally for efficiency.
      with tf.variable_scope('softmax_linear') as scope:
        reshape = tf.reshape(net, [inputs.get_shape()[0].value, -1])
        dim = reshape.get_shape()[1].value

        weights = _variable_with_weight_decay('weights', [dim, NUM_CLASSES],
                                              stddev=0.05, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear, None


def loss(logits, labels, batch_size=None, aux_loss=False):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  logits = logits[0]

  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection(LOSSES_COLLECTION, cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  # return tf.add_n(tf.get_collection(LOSSES_COLLECTION), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection(LOSSES_COLLECTION)
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
