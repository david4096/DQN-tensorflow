import gym
import random
import logging
import numpy as np
import tensorflow as tf
from threading import Thread

from dqn.agent import Agent
from dqn.network import DQN
from dqn.environment import GymEnvironment

flags = tf.app.flags

# Deep q Network
flags.DEFINE_string('DQN_type', 'nips', 'The type of DQN. [nature, nips]')
flags.DEFINE_string('data_format', 'NCHW', 'The format of convolutional filter. NHWC for CPU and NCHW for GPU')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of actions to repeat')
flags.DEFINE_integer('random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_integer('screen_height', 84, 'The height of gym screen')
flags.DEFINE_integer('screen_width', 84, 'The width of gym screen')
flags.DEFINE_integer('history_length', 4, 'The length of history of screens to use as an input to DQN')
flags.DEFINE_integer('max_reward', +1, 'The maximum value of clipped reward')
flags.DEFINE_integer('min_reward', -1, 'The minimum value of clipped reward')
flags.DEFINE_boolean('minus_one_if_dead', False, 'Whether to -1 to reward if a life is discounted')

# Timer
flags.DEFINE_integer('train_frequency', 4, '')

# Below numbers will be multiplied by scale
flags.DEFINE_integer('scale', 10000, 'The scale for big numbers')
flags.DEFINE_integer('memory_size', 100, 'The size of experience memory (*= scale)')
flags.DEFINE_integer('t_target_q_update_freq', 1, 'The frequency of target network to be updated (*= scale)')
flags.DEFINE_integer('t_test', 1, 'The maximum number of t while training (*= scale)')
flags.DEFINE_integer('t_ep_end', 100, 'The time when epsilon reach ep_end (*= scale)')
flags.DEFINE_integer('t_train_max', 5000, 'The maximum number of t while training (*= scale)')
flags.DEFINE_float('t_learn_start', 5, 'The time when to begin training (*= scale)')
flags.DEFINE_float('learning_rate_decay_step', 5, 'The learning rate of training (*= scale)')

# Training
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('max_delta', None, 'The maximum value of delta')
flags.DEFINE_integer('min_delta', None, 'The minimum value of delta')
flags.DEFINE_float('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_float('ep_end', 0.01, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_float('discount', 0.99, 'Discount factor of return')
#flags.DEFINE_integer('max_grad_norm', 40, 'The maximum gradient norm of RMSProp optimizer')

# Optimizer
flags.DEFINE_float('learning_rate', 25e-3, 'The learning rate of training')
flags.DEFINE_float('learning_rate_minimum', 25e-4, 'The minimum of learning rate')
flags.DEFINE_float('learning_rate_decay', 0.99, 'The rate of learning rate decay')

# Machine
flags.DEFINE_integer('n_worker', 4, 'The number of workers to run asynchronously')
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')

# Debug
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

config = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False

logger.setLevel(config.log_level)

# Set random seed
tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

if config.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print " [*] GPU : %.4f" % fraction
  return fraction

def main(_):
  if config.use_gpu:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(config.gpu_fraction))
  else:
    gpu_options = None
    config.cnn_format = 'NHWC'

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    global_t = np.array([0])
    action_size = gym.make(config.env_name).action_space.n

    global_network = DQN(config, action_size, sess)

    loss = tf.reduce_mean(tf.square(clipped_delta), name='loss')
    learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
    learning_rate_op = tf.maximum(learning_rate_minimum,
        tf.train.exponential_decay(
            learning_rate,
            learning_rate_step,
            learning_rate_decay_step,
            learning_rate_decay,
            staircase=True))

    global_step = tf.Variable(0, trainable=False)

    global_optim = tf.train.RMSPropOptimizer(
        learning_rate_op, momentum=0.95, epsilon=0.01).minimize(loss)

    agents = {}
    for worker_id in xrange(config.n_worker):
      with tf.variable_scope('thread%d' % worker_id) as scope:
        if config.env_type == 'simple':
          env = SimpleGymEnvironment(config)
        else:
          env = GymEnvironment(config)
        agents[worker_id] = Agent(config, env, global_network, sess)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

    if config.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
