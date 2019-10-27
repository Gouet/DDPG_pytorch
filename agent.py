import torch
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

class Actor(torch.nn.Module):
    def __init__(self, inputs, scaled, tau=0.001):
        super(Actor, self).__init__()

        self.scaled = scaled
        print('self.scaled:', self.scaled)
        self.tau = tau

        self._fc1 = torch.nn.Linear(inputs, 400)
        self._bn1 = torch.nn.BatchNorm1d(400)
        self._relu1 = torch.nn.ReLU(inplace=True)

        self._fc2 = torch.nn.Linear(400, 300)
        self._bn2 = torch.nn.BatchNorm1d(300)
        self._relu2 = torch.nn.ReLU(inplace=True)

        self._fc3 = torch.nn.Linear(300, 1)
        self._tanh3 = torch.nn.Tanh()

        self._fc1.weight.data = fanin_init(self._fc1.weight.data.size())      
        self._fc2.weight.data = fanin_init(self._fc2.weight.data.size())
        self._fc3.weight.data.uniform_(-0.003, 0.003)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inputs):
        fc1 = self._relu1(self._bn1(self._fc1(inputs)))
        fc2 = self._relu2(self._bn2(self._fc2(fc1)))
        output = self._tanh3(self._fc3(fc2))
        
        output = output * self.scaled
        return output

    def train_step(self, critic, states):
        actor_loss = -critic(states, self(states)).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

    def update(self, actor):
        for param, target_param in zip(actor.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_copy(self, actor):
        for param, target_param in zip(actor.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

class Critic(torch.nn.Module):
    def __init__(self, inputs, actions, tau=0.001):
        super(Critic, self).__init__()
        
        self.fc1 = torch.nn.Linear(inputs, 400)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.bn1 = torch.nn.BatchNorm1d(400)
        self.fc2 = torch.nn.Linear(400 + actions, 300)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        
        self.fc3 = torch.nn.Linear(300, 1)
        self.fc3.weight.data.uniform_(-0.003, 0.003)

        self.ReLU = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=0.001)

        self.tau = tau

    def forward(self, inputs, actions):
        h1 = self.ReLU(self.fc1(inputs))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(torch.cat([h1_norm, actions], dim=1)))
        Qval = self.fc3(h2)

        return Qval

    def train_step(self, states, actions, yi):
        current_Q = self(states, actions)

        critic_loss = F.mse_loss(current_Q, yi)

        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        return current_Q

    def update(self, critic):
        for param, target_param in zip(critic.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_copy(self, critic):
        for param, target_param in zip(critic.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
