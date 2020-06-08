import random, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque

# Training params
GAMMA = 0.9
min_hidden = 512
init_w=3e-3

class Policy(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden = min(min_hidden, int(self.state_size * 1.5))
        self.linear1 = nn.Linear(self.state_size, self.hidden)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

        self.linear3 = nn.Linear(self.hidden, self.num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.softmax(self.linear3(x), dim=-1)

    def get_action(self, state):
        state = state.float()
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

class REINFORCE_model:
    def __init__(self, state_size, num_actions):
        self.policy = Policy(state_size, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.log_probs = []
        self.rewards = []

    def update_policy(self):
        discounted_rewards = []

        for t in range(len(self.rewards)):
            Gt = 0
            pw = 0
            for r in self.rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_gradient = []
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()
        self.reset()

    def train(self):
        return

    def update_target(self):
        return

    def reset(self):
        self.log_probs = []
        self.rewards = []

    def get_action(self, state):
        action, log_prob = self.policy.get_action(state)
        self.log_probs.append(log_prob)
        return action

    def push_replay(self, state, action, reward, done, state_t1):
        self.rewards.append(reward)

    def save_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        torch.save({ "policy_state_dict": self.policy.state_dict(),
                   }, filename)

    def load_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            return True
        return False
