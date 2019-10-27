import gym
import time
import numpy as np
import ddpg
import os
import agent
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning parser for DDPG')

    parser.add_argument('--scenario', type=str, default='Pendulum-v0')
    parser.add_argument('--eval', action='store_false')

    parser.add_argument('--load-episode-saved', type=int, default=50)
    parser.add_argument('--saved-episode', type=int, default=50)

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-episode', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)

    return parser.parse_args()

try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

def main(arglist):
    env = gym.make(arglist.scenario)
    writer = SummaryWriter(log_dir='./logs/')

    critic = agent.Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    actor = agent.Actor(env.observation_space.shape[0], 2).to(device)
    target_critic = agent.Critic(env.observation_space.shape[0], env.action_space.shape[0], arglist.tau).to(device)
    target_actor = agent.Actor(env.observation_space.shape[0], 2, arglist.tau).to(device)
    
    actor.eval()
    critic.eval()
    target_actor.eval()
    target_critic.eval()

    ddpg_algo = ddpg.DDPG(actor, critic, target_actor, target_critic, arglist.gamma, arglist.batch_size, arglist.eval)
    ddpg_algo.load('./saved/actor_' + str(arglist.load_episode_saved), './saved/critic_' + str(arglist.load_episode_saved))

    for episode in range(arglist.max_episode):
        obs = env.reset()
        done = False
        j = 0
        ep_ave_max_q_value = 0
        total_reward = 0
        while not done:
            if not arglist.eval:
                env.render()
            
            action = ddpg_algo.act(obs)

            obs2, reward, done, info = env.step(action)
            total_reward += reward

            if arglist.eval:
                ep_ave_max_q_value += ddpg_algo.train(action, [reward], obs, obs2, [done])
            obs = obs2
            j += 1

        if arglist.eval and episode % arglist.saved_episode == 0 and episode > 0:
            critic.save_model('./saved/critic_' + str(episode))
            actor.save_model('./saved/actor_' + str(episode))

        if arglist.eval:
            print('average_max_q: ', ep_ave_max_q_value / float(j), 'reward: ', total_reward, 'episode:', episode)
            writer.add_scalar('Average_max_q', ep_ave_max_q_value / float(j), episode)
            writer.add_scalar('Reward', total_reward, episode)

    env.close()

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)