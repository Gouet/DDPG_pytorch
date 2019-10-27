from collections import deque
import random
import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch], dtype='float32')
        a_batch = np.array([_[1] for _ in batch], dtype='float32')
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch], dtype='float32')

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class DDPG:
    def __init__(self, actor, critic, target_actor, target_critic, gamma, batch_size, train_mode):

        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_mode = train_mode

        self.target_actor.hard_copy(actor)
        self.target_critic.hard_copy(critic)

        self.ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1,))
        self.buffer = ReplayBuffer(100000)

    def load(self, filename_actor, filename_critic):
        try:
            self.critic.load_model(filename_critic)
            self.actor.load_model(filename_actor)
        except Exception as e:
            print(e.__repr__)

    def act(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)

        noise = self.ou()
        noise = torch.FloatTensor(noise).unsqueeze(0).to(device)
        action = self.actor(state)

        if self.train_mode:
            action = action + noise
        action = action.cpu().detach().numpy()[0]
        return action

    def train(self, action, reward, state, state2, done):
        self.buffer.add(state, action, reward, done, state2)
        ep_ave_max_q_value = 0

        if self.buffer.size() > self.batch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

            s_batch = torch.FloatTensor(s_batch).to(device)
            a_batch = torch.FloatTensor(a_batch).to(device)
            r_batch = torch.FloatTensor(r_batch).to(device)
            t_batch = torch.FloatTensor(t_batch).to(device)
            s2_batch = torch.FloatTensor(s2_batch).to(device)

            target_action2 = self.target_actor(s2_batch)
            predicted_q_value = self.target_critic(s2_batch, target_action2)

            yi = r_batch + ((1 - t_batch) * self.gamma * predicted_q_value).detach()

            predictions = self.critic.train_step(s_batch, a_batch, yi)

            ep_ave_max_q_value = np.amax(predictions.cpu().detach().numpy())

            self.actor.train_step(self.critic, s_batch)

            self.target_actor.update(self.actor)
            self.target_critic.update(self.critic)
        
        return ep_ave_max_q_value