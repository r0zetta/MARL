import random, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Training params
batch_size = 16
buffer_max = 10000

gamma=0.99
mean_lambda=1e-3
std_lambda=1e-3
z_lambda=0.0
soft_tau=1e-2
hidden_size = 256
init_w=3e-3
log_std_min=-20
log_std_max=2

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Implementation of soft-actor-critic based on https://github.com/higgsfield/RL-Adventure-2
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class GN(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(GN, self).__init__()

        self.linear1 = nn.Linear(state_dim + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, state_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)

        action  = action.detach().cpu().numpy()
        return action[0]

class SAC_model:
    def __init__(self, state_dim, num_actions,
                 with_guesses=True,
                 reward_multiplier=0.1):
        self.reward_multiplier = reward_multiplier
        self.value_net = ValueNetwork(state_dim).to(device)
        self.target_value_net = ValueNetwork(state_dim).to(device)

        self.soft_q_net = SoftQNetwork(state_dim, num_actions).to(device)
        self.policy_net = PolicyNetwork(state_dim, num_actions).to(device)

        for target_param, param in zip(self.target_value_net.parameters(),
                                       self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.learning_rate = 3e-4
        self.value_lr  = self.learning_rate
        self.soft_q_lr = self.learning_rate
        self.policy_lr = self.learning_rate

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.with_guesses = with_guesses
        self.gp = 0.0
        if self.with_guesses == True:
            self.gn = GN(state_dim, num_actions)
            self.gn_fixed = GN(state_dim, num_actions)
            self.gn_fixed.load_state_dict(self.gn.state_dict())
            self.gn_fixed.eval()
            self.gn_optimizer = optim.Adam(self.gn.parameters(), lr=3e-4)

        self.buffer_max = buffer_max
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.buffer_max)


    def push_replay(self, state, action, reward, done, state_t1):
        self.replay_buffer.push(state, action, reward, state_t1, done)

    def get_action(self, state):
        return self.policy_net.get_action(state)

    def update_target(self):
        return

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        if self.with_guesses == True:
            with torch.no_grad():
                f_guesses = self.gn_fixed(state, new_action).detach()
            guesses = self.gn(state, new_action)
            g_r = [(guesses[i] - f_guesses[i]).pow(2).mean() for i in range(len(guesses))]
            reward = torch.FloatTensor([reward[i] + g_r[i] + self.gp for i in range(len(g_r))])
        reward = torch.FloatTensor([x * self.reward_multiplier for x in reward])

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda * log_std.pow(2).mean()
        z_loss    = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        if self.with_guesses == True:
            next_state = next_state.view(next_state.size(0), -1)
            gn_loss = (guesses - next_state).pow(2).mean()

            self.gn_optimizer.zero_grad()
            gn_loss.backward(retain_graph=True)
            self.gn_optimizer.step()
            for param in self.gn.parameters():
                param.grad.data.clamp_(-1, 1)

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(),
                                       self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

