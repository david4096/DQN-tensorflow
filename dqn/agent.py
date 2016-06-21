import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .ops import linear, conv2d
from .replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl

class Agent(object):
  def __init__(self, config, environment, global_network, global_optim, learning_rate_op, sess):
    self.sess = sess
    self.weight_dir = 'weights'

    self.env = environment
    self.history = History(self.config)

    self.learning_rate = config.learning_rate
    self.learning_rate_op = learning_rate_op

    if not config.async:
      self.memory = ReplayMemory(self.config, self.model_dir)
    else:
      self.memory = None

    self.build_dqn()

  def train(self, global_t):
    self.global_t = global_t

    # 0. Prepare training
    state, reward, terminal = self.env.new_random_game()
    self.observe(state, reward, terminal)

    while True:
      if global_t[0] > self.t_train_max:
        break

      # 1. Predict
      action = self.predict(state)
      # 2. Step
      state, reward, terminal = self.env.step(-1, is_training=True)
      # 3. Observe
      self.observe(state, reward, terminal)

      if terminal:
        self.env.new_random_game()

      global_t[0] += 1

  def train_with_log(self, global_t):
    self.global_t = global_t

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []

    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if global_t[0] > self.t_train_max:
        break

      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. predict
      action = self.predict(self.history.get())
      # 2. act
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe
      self.observe(screen, reward, action, terminal)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
      total_reward += reward

      if self.step >= self.learn_start:
        if self.step % self.test_step == self.test_step - 1:
          avg_reward = total_reward / self.test_step
          avg_loss = self.total_loss / self.update_count
          avg_q = self.total_q / self.update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
              % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

          if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            self.step_assign_op.eval({self.step_input: self.step + 1})
            self.save_model(self.step + 1)

            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': num_game,
                'episode.rewards': ep_rewards,
                'episode.actions': actions,
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.update()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def update(self):
    if self.memory == None:
    else:
      if self.memory.count < self.history_length:
        return
      else:
        s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

      t = time.time()
      if self.double_q:
        # Double Q-learning
        pred_action = self.q_action.eval({self.s_t: s_t_plus_1})

        q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
          self.target_s_t: s_t_plus_1,
          self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
        })
        target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
      else:
        q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

      _, q_t, loss = self.sess.run([self.optim, self.q, self.loss], {
        self.target_q_t: target_q_t,
        self.action: action,
        self.s_t: s_t,
        self.learning_rate_step: self.step,
      })

    if self.worker_id == 0:
      self.total_loss += loss
      self.total_q += q_t.mean()
      self.update_count += 1

  def build_dqn(self):
    # target network
    with tf.variable_scope('prediction'):
      self.pred_network = DQN(config, self.sess)

    # target network
    with tf.variable_scope('target'):
      self.target_network = DQN(config, self.sess)

    self.target_network.create_copy_op(self.pred_network)

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted
      self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

    if self.worker_id == 0:
      with tf.variable_scope('summary'):
        scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
            'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

        self.summary_placeholders = {}
        self.summary_ops = {}

        for tag in scalar_summary_tags:
          self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
          self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

        histogram_summary_tags = ['episode.rewards', 'episode.actions']

        for tag in histogram_summary_tags:
          self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
          self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

        self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    if not self.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print "="*30
      print " [%d] Best reward : %d" % (best_idx, best_reward)
      print "="*30

    if not self.display:
      self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
