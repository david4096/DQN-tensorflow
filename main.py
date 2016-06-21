import random
import tensorflow as tf
from threading import Thread

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.app.flags
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_boolean('save_weight', False, 'Save weight from pickle file')
flags.DEFINE_boolean('load_weight', False, 'Load weight from pickle file')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('minus_one_if_dead', False, 'Whether to -1 to reward if a life is discounted')
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_boolean('async', True, 'Use asynchronous update')
flags.DEFINE_integer('n_worker', 4, 'The number of workers to run asynchronously')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')
FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print " [*] GPU : %.4f" % fraction
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    global_network = DQN(config, sess)

    config = get_config(FLAGS) or FLAGS
    if FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

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

    if FLAGS.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
