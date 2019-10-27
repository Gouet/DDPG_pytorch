import gym
import time
import numpy as np
import ddpg
import os
import agent
import torch
from torch.utils.tensorboard import SummaryWriter

TRAIN_MODE = False
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

env = gym.make('Pendulum-v0')

critic = agent.Critic(3, 1).to(device)
actor = agent.Actor(3, 2).to(device)
target_critic = agent.Critic(3, 1, 0.001).to(device)
target_actor = agent.Actor(3, 2, 0.001).to(device)

actor.eval()
critic.eval()
target_actor.eval()
target_critic.eval()

try:
    critic.load_model('./saved/critic')
    actor.load_model('./saved/actor')
except Exception as e:
    print(e.__repr__)

target_actor.hard_copy(actor)
target_critic.hard_copy(critic)

ou = ddpg.OrnsteinUhlenbeckActionNoise(mu=np.zeros(1,))
buffer = ddpg.ReplayBuffer(100000)
global ep_ave_max_q_value
ep_ave_max_q_value = 0
global total_reward
total_reward = 0

if TRAIN_MODE:
    writer = SummaryWriter(log_dir='./logs/')

def train(action, reward, state, state2, done):
    global ep_ave_max_q_value
    
    buffer.add(state, action, reward, done, state2)
    batch_size = 64

    if buffer.size() > batch_size:
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer.sample_batch(batch_size)

        s_batch = torch.FloatTensor(s_batch).to(device)
        a_batch = torch.FloatTensor(a_batch).to(device)
        r_batch = torch.FloatTensor(r_batch).to(device)
        t_batch = torch.FloatTensor(t_batch).to(device)
        s2_batch = torch.FloatTensor(s2_batch).to(device)

        target_action2 = target_actor(s2_batch)
        predicted_q_value = target_critic(s2_batch, target_action2)

        yi = r_batch + ((1 - t_batch) * 0.99 * predicted_q_value).detach()

        predictions = critic.train_step(s_batch, a_batch, yi)

        ep_ave_max_q_value += np.amax(predictions.cpu().detach().numpy())

        actor.train_step(critic, s_batch)

        target_actor.update(actor)
        target_critic.update(critic)

for episode in range(10000):
    obs = env.reset()
    done = False
    j = 0
    ep_ave_max_q_value = 0
    total_reward = 0
    while not done:
        if not TRAIN_MODE:
            env.render()
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        noise = ou()
        noise = torch.FloatTensor(noise).unsqueeze(0).to(device)
        action = actor(state)


        if TRAIN_MODE:
            action = action + noise

        action = action.cpu().detach().numpy()[0]

        obs2, reward, done, info = env.step(action)
        total_reward += reward

        if TRAIN_MODE:
            train(action, [reward], obs, obs2, [done])
        obs = obs2
        j += 1

    if TRAIN_MODE:
        critic.save_model('./saved/critic')
        actor.save_model('./saved/actor')

    if TRAIN_MODE:
        print('average_max_q: ', ep_ave_max_q_value / float(j), 'reward: ', total_reward, 'episode:', episode)
        writer.add_scalar('Average_max_q', ep_ave_max_q_value / float(j), episode)
        writer.add_scalar('Reward', total_reward, episode)

env.close()