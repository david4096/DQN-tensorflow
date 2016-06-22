import tensorflow as tf

from .ops import *

class DQN(object):
  def __init__(self, config, output_size, sess):
    self.var = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    if config.data_format == 'NHWC':
      self.s_t = tf.placeholder('float32',
          [None, config.screen_width, config.screen_height, config.history_length], name='s_t')
    else:
      self.s_t = tf.placeholder('float32',
          [None, config.history_length, config.screen_width, config.screen_height], name='s_t')

    self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.s_t,
        32, [8, 8], [4, 4], initializer, activation_fn, config.data_format, name='l1')
    self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
        64, [4, 4], [2, 2], initializer, activation_fn, config.data_format, name='l2')
    self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2,
        64, [3, 3], [1, 1], initializer, activation_fn, config.data_format, name='l3')

    shape = self.l3.get_shape().as_list()
    self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

    if config.dueling:
      self.value_hid, self.var['l4_val_w'], self.var['l4_val_b'] = \
          linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

      self.adv_hid, self.var['l4_adv_w'], self.var['l4_adv_b'] = \
          linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

      self.value, self.var['val_w_out'], self.var['val_w_b'] = \
        linear(self.value_hid, 1, name='value_out')

      self.advantage, self.var['adv_w_out'], self.var['adv_w_b'] = \
        linear(self.adv_hid, output_size, name='adv_out')

      # Average Dueling
      self.q = self.value + (self.advantage - 
        tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
    else:
      self.l4, self.var['l4_w'], self.var['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
      self.q, self.var['q_w'], self.var['q_b'] = linear(self.l4, output_size, name='q')

    self.q_action = tf.argmax(self.q, dimension=1)
    self.q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
    self.q_with_idx = tf.gather_nd(self.q, self.q_idx)

  def run_copy(self):
    if self.copy_op is None:
      raise Exception("run `create_copy_op` first before copy")
    else:
      self.sess.run(self.copy_op)

  def create_copy_op(self, network):
    with tf.variable_scope('copy_from_target'):
      copy_ops = []

      for name in self.var.keys():
        copy_op = self.var[name].assign(network.var[name])
        copy_ops.append(copy_op)

      self.copy_op = tf.group(*copy_ops, name='copy_op')
