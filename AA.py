import random, json, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Training params
init_w=3e-3

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4):
        super().__init__()
        self.k, self.heads = k, heads

        self.tokeys    = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues  = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x)   .view(b, t, h, k)
        values  = self.tovalues(x) .view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
          nn.Linear(k, 2 * k),
          nn.ReLU(),
          nn.Linear(2 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)

class AA(nn.Module):
    def __init__(self,  state_len,  num_actions, num_frames, transformer_depth=4,
                 GRU_hidden=32, GRU_layers=2, attention_heads=4, dropout=0.2):
        super(AA, self).__init__()
        self.state_len = state_len
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.GRU_hidden = GRU_hidden
        self.GRU_layers = GRU_layers
        self.attention_heads = attention_heads

        self.attention = SelfAttention(state_len, heads=attention_heads)

        self.gru1 = nn.GRU(self.state_len, self.GRU_hidden, self.GRU_layers)
        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(self.GRU_hidden, self.num_actions)
        self.head.weight.data.uniform_(-init_w, init_w)
        self.head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, hidden):
        x = self.attention(state)
        x, h = self.gru1(x, hidden)
        x = x[:, -1]
        x = self.drop(x)
        return F.softmax(self.head(x), dim=-1), h

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.GRU_layers, self.num_frames, self.GRU_hidden).zero_()
        return hidden

class AA_model:
    def __init__(self, state_len, num_actions, num_frames,
                 batch_size=1, reward_multiplier=1.0):
        self.reward_multiplier = reward_multiplier
        self.policy = AA(state_len, num_actions, num_frames)
        self.policy_hidden = self.policy.init_hidden()
        self.target = AA(state_len, num_actions, num_frames)
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
        Q_sa = Q_sa.detach()
        gamma = 0.99
        new_targets = torch.Tensor([reward_batch[n] + ((1 - done_t[n]) * (gamma * torch.max(Q_sa[n]))) for n in range(len(targets))])

        #policy_loss = F.smooth_l1_loss(targets, new_targets)
        policy_loss = F.mse_loss(targets, new_targets)

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
