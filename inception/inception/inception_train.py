# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A library to train Inception using multiple GPU's with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from inception import image_processing
from inception import inception_model as inception
from inception import alexnet_model as alexnet
from inception import cifar10
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

tf.app.flags.DEFINE_string('model_name', 'inception',
                           'inception (default), alexnet, cifar10.')

tf.app.flags.DEFINE_integer('num_examples_per_task', 64,
                            'examples to to fit in each gpu.')

tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

tf.app.flags.DEFINE_boolean('timeline', False,
                            'Collect timeline.')

tf.app.flags.DEFINE_boolean('save_graph', False,
                            'Save graph to tensorboard event file.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
  """Calculate the total loss on a single tower running the ImageNet model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = not FLAGS.fine_tune

  # Choose model based on flags.
  if FLAGS.model_name == 'inception':
    model = inception
  elif FLAGS.model_name == 'alexnet':
    model = alexnet
  elif FLAGS.model_name == 'cifar10':
    model = cifar10

  # Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    logits = model.inference(images, num_classes, for_training=True,
                             restore_logits=restore_logits,
                             scope=scope)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  model.loss(logits, labels, batch_size=split_batch_size)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(loss_name +' (raw)', l)
    tf.summary.scalar(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def _average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(dataset):
  """Train on dataset for a number of steps."""
  # Choose model based on flags.
  if FLAGS.model_name == 'inception':
    model = inception
  elif FLAGS.model_name == 'alexnet':
    model = alexnet
  elif FLAGS.model_name == 'cifar10':
    model = cifar10

  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, 
                     slim.variables.VARIABLES_TO_RESTORE])

    num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                             FLAGS.batch_size)
    if FLAGS.model_name == 'cifar10':
      # Calculate the learning rate schedule.
      decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      cifar10.LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)
      opt = tf.train.GradientDescentOptimizer(lr)
    else:
      # Calculate the learning rate schedule.
      decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)

      # Create an optimizer that performs gradient descent.
      opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                      momentum=RMSPROP_MOMENTUM,
                                      epsilon=RMSPROP_EPSILON)

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')
    split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

    assert split_batch_size % FLAGS.num_examples_per_task == 0, (
        'Batch size must be divisible by number of GPUs * batch size per gpu')
    tasks_per_gpu = split_batch_size // FLAGS.num_examples_per_task
    num_tasks = FLAGS.batch_size // FLAGS.num_examples_per_task

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus

    if FLAGS.model_name in ['inception', 'alexnet']:
      # Use ImageNet.
      images, labels = image_processing.distorted_inputs(
          dataset,
          num_preprocess_threads=num_preprocess_threads)
    else:
      # Use CIFAR.
      images, labels = cifar10.distorted_inputs()

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    if FLAGS.model_name == 'cifar10':
      num_classes = dataset.num_classes()
    else:
      num_classes = dataset.num_classes() + 1

    # Split the batch of images and labels for towers.
    images_splits = tf.split(axis=0, num_or_size_splits=num_tasks, value=images)
    labels_splits = tf.split(axis=0, num_or_size_splits=num_tasks, value=labels)

    # Calculate the gradients for each model tower.
    tower_grads = []
    losses = []

    def ops_one_tower(i, t, reuse_variables, scope):
      task_images = images_splits[i * tasks_per_gpu + t]
      task_labels = labels_splits[i * tasks_per_gpu + t]

      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        # Calculate the loss for one tower of the ImageNet model. This
        # function constructs the entire ImageNet model but shares the
        # variables across all towers.
        loss = _tower_loss(task_images, task_labels, num_classes,
                           scope, reuse_variables)
        losses.append(loss)

      # Reuse variables for the next tower.
      # reuse_variables = True

      # Retain the summaries from the final tower.
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

      # Retain the Batch Normalization updates operations only from the
      # final tower. Ideally, we should grab the updates from all towers
      # but these stats accumulate extremely fast so we can ignore the
      # other stats from the other towers without significant detriment.
      batchnorm_updates = tf.get_collection(
        slim.ops.UPDATE_OPS_COLLECTION,
        scope)

      # Calculate the gradients for the batch of data on this ImageNet
      # tower.
      grads = opt.compute_gradients(loss)

      # Keep track of the gradients across all towers.
      tower_grads.append(grads)

      return grads[0], summaries, batchnorm_updates

    reuse_variables = None
    summaries = None
    batchnorm_updates = None
    for i in range(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        task_dep = [tf.constant(0)]
        for t in range(tasks_per_gpu):
          with tf.name_scope('%s_%d_%d' % (model.TOWER_NAME, i, t)) as scope:
            with tf.control_dependencies(task_dep):
              task_dep, summaries, batchnorm_updates = ops_one_tower(
                i, t, reuse_variables, scope)
              reuse_variables = True

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = _average_gradients(tower_grads)

    # Add a summaries for the input processing and global_step.
    summaries.extend(input_summaries)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))
    summaries.append(tf.summary.scalar('global_step', global_step))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY, global_step)

    # Another possibility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all updates to into a single train op.
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()


    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    tf.logging.info('Starting session.')
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    step = 0
    if FLAGS.pretrained_model_checkpoint_path:
      # assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)

      # Here we need to restore all variables, including the shadow moving average
      # ones. Note that we remove tower-specific ones, as the true variables are
      # the CPU shared by towers.
      #   variables_to_restore = tf.get_collection(
      #       slim.variables.VARIABLES_TO_RESTORE)
      #   restorer = tf.train.Saver(variables_to_restore)
      variables_to_restore = tf.global_variables()
      variables_to_restore = list(
        filter(lambda x: x.name[:5] != 'tower', variables_to_restore))
      restorer = tf.train.Saver(variables_to_restore)

      restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)

      step = int(sess.run(global_step))
      print('%s: Successfully loaded model from %s at step=%d.' %
            (datetime.now(), FLAGS.pretrained_model_checkpoint_path, step))
      step += 1

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir,
        graph=(sess.graph if FLAGS.save_graph else None))

    run_metadata = tf.RunMetadata()
    run_options = None
    if FLAGS.timeline:
      run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    # Train and concurrently run the summary operation at a
    # specified interval.
    tf.logging.info('Start training loop.')
    next_summary_time = time.time() + FLAGS.save_summaries_secs
    while step < FLAGS.max_steps:
      start_time = time.time()
      ret = sess.run([train_op] + losses,
                     options=run_options,
                     run_metadata=run_metadata)
      loss_value = np.mean(ret[1:])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      #if step % 10 == 0:
      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print(format_str % (datetime.now(), step, loss_value,
                          examples_per_sec, duration))

      # Output timeline for tracing.
      if FLAGS.timeline and step % 30 == 0:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open(os.path.join(
                  FLAGS.train_dir, '..',
                  'timeline.ctf_%d.json' % (step)),
                  'w') as trace_file:
          trace_file.write(trace.generate_chrome_trace_format())

      if step + 1 == FLAGS.max_steps or  next_summary_time < time.time():
        tf.logging.info('Running Summary operation.')
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        tf.logging.info('Finished running Summary operation.')

        # Determine the next time for running the summary.
        next_summary_time += FLAGS.save_summaries_secs

      # Save the model checkpoint periodically.
      if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      
      step += 1
