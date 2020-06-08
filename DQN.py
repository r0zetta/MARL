import random, os, sys, json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Training params
gamma = 0.99
min_hidden = 512
init_w=3e-3

class GN(nn.Module):
    def __init__(self, state_size, num_actions):
        super(GN, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden = min(min_hidden, int(self.state_size * 1.5))
        self.linear1 = nn.Linear(self.state_size + self.num_actions, self.hidden)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

        self.linear3 = nn.Linear(self.hidden, self.state_size)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, actions):
        x = torch.cat([state, actions], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class DQN(nn.Module):
    def __init__(self, state_size, num_actions, with_softmax=False, dropout=0.0):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions
        self.with_softmax = with_softmax
        self.hidden = min(min_hidden, int(self.state_size * 1.5))
        self.linear1 = nn.Linear(self.state_size, self.hidden)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

        self.drop = nn.Dropout(dropout)

        self.linear3 = nn.Linear(self.hidden, self.num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.drop(x)
        if self.with_softmax == True:
            return F.softmax(self.linear3(x), dim=-1)
        else:
            return self.linear3(x)

class DQN_model:
    def __init__(self, state_size, num_actions,
                 with_guesses=False, sequential_replay=False,
                 with_softmax=False, batch_size=64,
                 buffer_max=10000, reward_multiplier=1.0):
        self.with_guesses = with_guesses
        self.with_softmax = with_softmax
        self.sequential_replay = sequential_replay
        self.reward_multiplier = reward_multiplier

        self.policy = DQN(state_size, num_actions, with_softmax=self.with_softmax)
        self.target = DQN(state_size, num_actions, with_softmax=self.with_softmax)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.policy_optimizer = optim.RMSprop(self.policy.parameters(), lr=3e-4)

        self.gp = 0.0
        if with_guesses == True:
            self.gn = GN(state_size, num_actions)
            self.gn_fixed = GN(state_size, num_actions)
            self.gn_fixed.load_state_dict(self.gn.state_dict())
            self.gn_fixed.eval()
            self.gn_optimizer = optim.Adam(self.gn.parameters(), lr=3e-4)

        self.replay_buffer = deque()
        self.buffer_max = buffer_max
        self.batch_size = batch_size

    def train(self):
        if len(self.replay_buffer) < self.batch_size + 1:
            return
        if self.sequential_replay == True:
            startx = random.randint(0, len(self.replay_buffer)-self.batch_size)
            minibatch = list(self.replay_buffer)[startx:startx+self.batch_size]
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_t, action_t, reward_t, done_t, state_t1 = zip(*minibatch)
        state_t = torch.cat(state_t)
        state_t1 = torch.cat(state_t1)
        action_batch = torch.LongTensor(action_t).unsqueeze(1)
        targets = self.policy(state_t)

        if self.with_guesses == True:
            with torch.no_grad():
                f_states = self.gn_fixed(state_t, targets)
                f_states.detach()

            g_states = self.gn(state_t, targets)
            g_sr = [(g_states[i] - f_states[i]).pow(2).mean() for i in range(len(g_states))]
            reward_batch = torch.FloatTensor([reward_t[i] + g_sr[i] + self.gp for i in range(len(g_sr))])
        else:
            reward_batch = torch.FloatTensor(reward_t)
        reward_batch = torch.FloatTensor([x * self.reward_multiplier for x in reward_batch])

        targets = targets.gather(1, action_batch).squeeze(1)
        Q_sa = self.target(state_t1).detach()
        new_targets = torch.Tensor([reward_batch[n] + ((1-done_t[n]) * (gamma * torch.max(Q_sa[n]))) for n in range(len(targets))])

        policy_loss = F.smooth_l1_loss(targets, new_targets)

        if self.with_guesses == True:
            gn_loss = (g_states - state_t1.view(state_t1.size(0), -1)).pow(2).mean()

            self.gn_optimizer.zero_grad()
            gn_loss.backward(retain_graph=True)
            self.gn_optimizer.step()
            for param in self.gn.parameters():
                param.grad.data.clamp_(-1, 1)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def get_action(self, state):
        with torch.no_grad():
            pred = self.policy(state)
            return pred.detach().numpy()[0]

    def get_actions(self, states):
        with torch.no_grad():
            state_t = torch.cat(states)
            targets = self.policy(state_t)
            return targets.detach().numpy()

    def push_replay(self, state, action, reward, done, state_t1):
        self.replay_buffer.append([state, action, reward, done, state_t1])
        if len(self.replay_buffer) > self.buffer_max:
            self.replay_buffer.popleft()

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
