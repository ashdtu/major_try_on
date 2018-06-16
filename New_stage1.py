
""" Stage2: given control point generate warpped images and use it for refinement.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import time

from utils import *

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tps_transformer import tps_stn

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",
                       "./prepare_data/tfrecord/zalando-train-?????-of-00032",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("mode", "train", "Training or testing")
tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
tf.flags.DEFINE_string("gen_checkpoint", "",
                       "Checkpoint path to the initial generative model.")
tf.flags.DEFINE_string("output_dir", "model/stage2/",
                       "Output directory of images.")
tf.flags.DEFINE_string("vgg_model_path", "./model/imagenet-vgg-verydeep-19.mat",
                       "model of the trained vgg net.")

tf.flags.DEFINE_integer("number_of_steps", 100000,
                        "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("batch_size", 16, "Size of mini batch.")
tf.flags.DEFINE_integer("num_preprocess_threads", 1, "")
tf.flags.DEFINE_integer("values_per_input_shard", 433, "")
tf.flags.DEFINE_integer("ngf", 64,
                        "number of generator filters in first conv layer")
tf.flags.DEFINE_integer("ndf", 64,
"number of discriminator filters in first conv layer")

tf.flags.DEFINE_integer("summary_freq", 50, #100
                        "update summaries every summary_freq steps")
tf.flags.DEFINE_integer("progress_freq", 10, #100
                        "display progress every progress_freq steps")
tf.flags.DEFINE_integer("trace_freq", 0,
                        "trace execution every trace_freq steps")
tf.flags.DEFINE_integer("display_freq", 300, #300
                        "write current training images every display_freq steps")
tf.flags.DEFINE_integer("save_freq", 1000,
                        "save model every save_freq steps, 0 to disable")

tf.flags.DEFINE_float("number_of_samples", 14221.0, "Samples in training set.")
tf.flags.DEFINE_float("lr", 0.0002, "Initial learning rate.")
tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
tf.flags.DEFINE_float("content_l1_weight", 0.2, "Weight on L1 term of content.")
tf.flags.DEFINE_float("perceptual_weight", 0.8, "weight on GAN term.")
tf.flags.DEFINE_float("tv_weight", 0.000005, "weight on TV term.")
tf.flags.DEFINE_float("mask_weight", 0.1, "weight on the selection mask.")


tf.logging.set_verbosity(tf.logging.INFO)

Model = collections.namedtuple("Model",
                               "gen_image_outputs, stn_image_outputs,"
                               "image_outputs, prod_mask_outputs,"
                               "mask_loss, tv_loss,"
                               "gen_loss_GAN, select_mask,"
                               "gen_loss_content_L1, perceptual_loss,"
                               "train, global_step")

FINAL_HEIGHT = 256
FINAL_WIDTH = 192

def is_training():
  return FLAGS.mode == "train"

def create_generator(product_image, body_seg, skin_seg,
                     pose_map, generator_outputs_channels):
  """ Generator from product images, segs, poses to a segment map"""
  # Build inputs
  generator_inputs = tf.concat([product_image, body_seg, skin_seg, pose_map],
                               axis=-1)
  # generator_inputs = tf.concat([body_seg, skin_seg, pose_map],
  #                              axis=-1)

  layers = []

  # encoder_1: [batch, 256, 192, in_channels] => [batch, 128, 96, ngf]
  with tf.variable_scope("encoder_1"):
    output = conv(generator_inputs, FLAGS.ngf, stride=2)
    layers.append(output)

  layer_specs = [
      # encoder_2: [batch, 128, 96, ngf] => [batch, 64, 48, ngf * 2]
      FLAGS.ngf * 2,
      # encoder_3: [batch, 64, 48, ngf * 2] => [batch, 32, 24, ngf * 4]
      FLAGS.ngf * 4,
      # encoder_4: [batch, 32, 24, ngf * 4] => [batch, 16, 12, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_5: [batch, 16, 12, ngf * 8] => [batch, 8, 6, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_6: [batch, 8, 6, ngf * 8] => [batch, 4, 3, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_7: [batch, 4, 3, ngf * 8] => [batch, 2, 1, ngf * 8]
      # FLAGS.ngf * 8,
  ]

  for out_channels in layer_specs:
    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
      rectified = lrelu(layers[-1], 0.2)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height/2, in_width/2, out_channels]
      convolved = conv(rectified, out_channels, stride=2)
      output = batch_norm(convolved, is_training())
      layers.append(output)


  layer_specs = [
      # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
      # (FLAGS.ngf * 8, 0.5),
      # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
      # (FLAGS.ngf * 8, 0.5),
      # decoder_6: [batch, 4, 3, ngf * 8 * 2] => [batch, 8, 6, ngf * 8 * 2]
      (FLAGS.ngf * 8, 0.5),
      # decoder_5: [batch, 8, 12, ngf * 8 * 2] => [batch, 16, 12, ngf * 8 * 2]
      (FLAGS.ngf * 8, 0.0),
      # decoder_4: [batch, 16, 12, ngf * 8 * 2] => [batch, 32, 24, ngf * 4 * 2]
      (FLAGS.ngf * 4, 0.0),
      # decoder_3: [batch, 32, 24, ngf * 4 * 2] => [batch, 64, 48, ngf * 2 * 2]
      (FLAGS.ngf * 2, 0.0),
      # decoder_2: [batch, 64, 48, ngf * 2 * 2] => [batch, 128, 96, ngf * 2]
      (FLAGS.ngf, 0.0),
  ]

  num_encoder_layers = len(layers)
  for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
    skip_layer = num_encoder_layers - decoder_layer - 1
    with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
      if decoder_layer == 0:
        # first decoder layer doesn't have skip connections
        # since it is directly connected to the skip_layer
        input = layers[-1]
      else:
        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

      rectified = tf.nn.relu(input)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height*2, in_width*2, out_channels]
      output = deconv(rectified, out_channels)
      output = batch_norm(output, is_training())

      if dropout > 0.0 and is_training():
        output = tf.nn.dropout(output, keep_prob=1 - dropout)

      layers.append(output)

  # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256,
  # generator_outputs_channels]
  with tf.variable_scope("decoder_1"):
    input = tf.concat([layers[-1], layers[0]], axis=3)
    rectified = tf.nn.relu(input)
    output = deconv(rectified, generator_outputs_channels)
    output = tf.tanh(output)
    layers.append(output)

return layers[-1]

PUT THE NEW MODEL FROM HERE
