import random, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Training params
init_w=3e-3
gamma = 0.99

class DRQN(nn.Module):
    def __init__(self,  state_len,  num_actions, num_frames,
                 GRU_hidden=128, GRU_layers=1, dropout=0.5):
        super(DRQN, self).__init__()
        self.state_len = state_len
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.GRU_hidden = GRU_hidden
        self.GRU_layers = GRU_layers
        self.gru1 = nn.GRU(self.state_len,
                           self.GRU_hidden,
                           self.GRU_layers,
                           bidirectional=False)

        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(self.GRU_hidden, self.num_actions)
        self.head.weight.data.uniform_(-init_w, init_w)
        self.head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, hidden):
        x, h = self.gru1(state, hidden)
        x = x[:, -1]
        x = self.drop(x)
        x = F.softmax(self.head(x), dim=-1), h
        return x

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.GRU_layers, self.num_frames, self.GRU_hidden).zero_()
        return hidden

class DRQN_model:
    def __init__(self, state_len, num_actions, num_frames,
                 batch_size=1, reward_multiplier=1.0):
        self.reward_multiplier = reward_multiplier
        self.policy = DRQN(state_len, num_actions, num_frames)
        self.policy_hidden = self.policy.init_hidden()
        self.target = DRQN(state_len, num_actions, num_frames)
        self.target_hidden = self.target.init_hidden()
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.replay_buffer = deque()
        self.buffer_max = batch_size
        self.batch_size = batch_size

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = list(self.replay_buffer)
        state_t, action_t, reward_t, done_t, state_t1 = zip(*minibatch)

        state_t = torch.cat(state_t)
        state_t1 = torch.cat(state_t1)
        action_batch = torch.LongTensor(action_t).unsqueeze(1)
        targets, phidden = self.policy(state_t, self.policy_hidden)
        self.policy_hidden = phidden

        reward_batch = torch.FloatTensor([x * self.reward_multiplier for x in reward_t])

        targets = targets.gather(1, action_batch).squeeze(1)
        Q_sa, thidden = self.target(state_t1, self.target_hidden)
        Q_sa_d = Q_sa.detach()
        new_targets = torch.Tensor([reward_batch[n] + ((1 - done_t[n]) * (gamma * torch.max(Q_sa_d[n]))) for n in range(len(targets))])

        policy_loss = F.smooth_l1_loss(targets, new_targets)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def get_action(self, state):
        with torch.no_grad():
            pred, hidden = self.policy(state, self.policy_hidden)
            self.policy_hidden = hidden
            return pred.detach().numpy()[0]

    def push_replay(self, state, action, reward, done, state_t1):
        self.replay_buffer.append([state, action, reward, done, state_t1])
        if len(self.replay_buffer) > self.buffer_max:
            self.replay_buffer.popleft()

    def save_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        torch.save({ "policy_state_dict": self.policy.state_dict(),
                     "policy_hidden": self.policy_hidden,
                   }, filename)

    def load_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_hidden = checkpoint["policy_hidden"]
            return True
        return False
