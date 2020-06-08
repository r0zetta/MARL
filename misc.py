import numpy as np
import torch
from DRQN import *
from DQN import *
from REINFORCE import *
from SAC import *
from AA import *
from PG import *

visible = 4
epsilon = 1.0
update_iter = 1000

# Models - naming is somewhat arbitrary
# DQN is simple feedforward nn implementing the basic DQN algo
# DRQN is same as DQN, but uses an RNN
# AA is same as DQN, but using an attention mechanism
# SAC implements Soft Actor Critic
# REINFORCE is a policy-gradient-based approach using a feedforward network
# PG is the same as REINFORCE, but uses an RNN
all_models = ["DQN", "DRQN", "SAC", "AA", "REINFORCE", "PG"]
sac_models = ["SAC"]
no_epsilon = ["SAC", "REINFORCE", "PG"]
flat_models = ["DQN", "SAC", "REINFORCE"]
train_after_episode = ["REINFORCE", "PG"]
rnn_models = ["DRQN", "AA", "PG"]

# Console colors
red = "\x1b[0;31;40m"
gre = "\x1b[0;32;40m"
yel = "\x1b[0;33;40m"
whi = "\x1b[0;37;40m"
cya = "\x1b[0;36;40m"
blu = "\x1b[0;34;40m"
pur = "\x1b[0;35;40m"
end = "\x1b[0m"

# Detailed printout of model predictions
def print_agent_details(gs, index):
    x, y, s = gs.agents[index]
    visible = gs.get_visible(x, y)
    msg = "\n"
    for column in visible:
        for item in column:
            msg += gs.get_printable(item)
        msg += "\n"
    print(msg)

def pretty_pred(move):
    color = 31 + move
    return "\x1b[1;"+str(31+move)+";40m" + str(move) + end

def pretty_preds(preds):
    return "[" + "".join([pretty_pred(x) for x in preds]) + "]"

def pretty_vecs(vecs):
    if len(vecs.shape) > 1:
        vecs = vecs[-1]
    top = np.argmax(vecs)
    normal = whi
    highlight = red
    vs = []
    for i, v in enumerate(vecs):
        start = normal
        e = ""
        if v >= 0:
            e = " "
        if i == top:
            start = highlight
        vs.append(start + e + "%.2f"%v + end)
    sep = normal + ", " + end
    ret = "[" + sep.join(vs) + "]"
    return ret

def get_state(gs, index, model_type):
    state = torch.FloatTensor(gs.get_agent_state(index))
    if model_type not in sac_models:
        state = state.unsqueeze(0)
    return state

def get_state_params(model_type):
    flatten_state = False
    if model_type in flat_models:
        prev_states = 1
        flatten_state = True
    elif model_type in rnn_models:
        prev_states = 4
    return flatten_state, prev_states

def get_model(model_type, gs):
    num_actions = gs.num_actions
    if model_type == "DQN":
        state_size = gs.get_agent_state(0).shape[0]
        model = DQN_model(state_size, num_actions, with_guesses=False, with_softmax=True)
    elif model_type == "DRQN":
        num_frames = gs.get_agent_state(0).shape[0]
        state_size = gs.get_agent_state(0).shape[1]
        model = DRQN_model(state_size, num_actions, num_frames)
    elif model_type == "REINFORCE":
        state_size = gs.get_agent_state(0).shape[0]
        model = REINFORCE_model(state_size, num_actions)
    elif model_type == "SAC":
        state_size = gs.get_agent_state(0).shape[0]
        model = SAC_model(state_size, num_actions)
    elif model_type == "AA":
        num_frames = gs.get_agent_state(0).shape[0]
        state_size = gs.get_agent_state(0).shape[1]
        model = AA_model(state_size, num_actions, num_frames)
    elif model_type == "PG":
        num_frames = gs.get_agent_state(0).shape[0]
        state_size = gs.get_agent_state(0).shape[1]
        model = PG_model(state_size, num_actions, num_frames)

    return model
