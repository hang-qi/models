from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from inception import inception_train
from inception.imagenet_data import ImagenetData
from inception.cifar10_data import Cifar10Data
from inception import cifar10

FLAGS = tf.app.flags.FLAGS


def main(_):
  if FLAGS.model_name == 'cifar':
    dataset = Cifar10Data(subset=FLAGS.subset)
    cifar10.maybe_download_and_extract()
  else:
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()

  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  inception_train.train(dataset)


if __name__ == '__main__':
  tf.app.run()
