from av_game import *
from misc import *
import time, sys, os, re, json
from collections import Counter

# Adjust the pvp game
# each team starts in a corner near a stationary boss surrounded by walls
# rewards are only given when an agent successfully attacks the other team's boss
# powerups exist on the battlefield that give agents more attack or hp
# agents can melee other team members
# if an agent is killed, it respawns at its original point
# agents can attack walls until they are destroyed

preds_len = 30
episode_limit = 4096
epsilon_degrade = 0.00001
min_epsilon = 0
model_type = "PG"
with_guesses = False
with_softmax = True
print_visuals = True
print_preds = False

def print_all_preds(all_preds):
    msg = ""
    for index, pred in enumerate(all_preds[:preds_len]):
        x, y, t, u, h = gs.agents[index]
        first = red + "dead" + end
        hp = ""
        rew = ""
        if index in got_rewards:
            rew = " " + gre + "+1" + end
        if index in did_bad:
            rew = " " + red + "-1" + end
        if h > 9:
            hp = cya + "%02d"%h + end
        elif h > 4:
            hp = pur + "%02d"%h + end
        else:
            hp = red + "%02d"%h + end
        if h > 0:
            if index not in stochastic:
                if s == 0:
                    first = "\x1b[6;30;42m" + "pred" + end
                else:
                    first = "\x1b[6;30;43m" + "pred" + end
            else:
                if s == 0:
                    first = gre + "stoc" + end
                else:
                    first = yel + "stoc" + end
            move = np.argmax(pred)
            plist[index].append(move)
            if len(plist[index]) > 30:
                plist[index].popleft()
        lpred = pretty_vecs(last_pred[index]) + " "
        ppred = pretty_preds(plist[index]) + " "
        msg += first + "("+"%02d"%index+"):" + ppred + lpred + hp + rew + "\n" 
    print(msg)

dirname = model_type + "_av_game_save"

# Get indices of all currently saved models
trained_models = []
if not os.path.exists(dirname):
    os.makedirs(dirname)
else:
    fns = os.listdir(dirname)
    for fn in fns:
        m = re.match(".+\_([0-9]+)\.pt$", fn)
        if m is not None:
            model_num = m.group(1)
            trained_models.append(int(model_num))

#random.seed(1)
flatten_state, prev_states = get_state_params(model_type)
gs = game_space(visible=visible, prev_states=prev_states,
                flatten_state=flatten_state, split_layers=True)

models = []
for n in range(gs.num_agents):
    model = get_model(model_type, gs)
    if n in trained_models:
        if model.load_model(dirname, n) == True:
            model.update_target()
    else:
        if len(trained_models) > 0:
            rindex = random.choice(trained_models)
            if model.load_model(dirname, rindex) == True:
                model.update_target()
    models.append(model)

tracked_rewards = []
filename = os.path.join(dirname, "rewards.json")
if os.path.exists(filename):
    with open(filename, "r") as f:
        for line in f:
            tracked_rewards = json.loads(line)
            break

tracked_durations = []
filename = os.path.join(dirname, "durations.json")
if os.path.exists(filename):
    with open(filename, "r") as f:
        for line in f:
            tracked_durations = json.loads(line)
            break

total_rewards = 0
episode = 1

if len(tracked_rewards) > 0:
    total_rewards = sum(tracked_rewards)
    episode = len(tracked_rewards)

episode_rewards = 0
current_iterations = 0
last_iterations = 0
iteration = 0
last_winner = None
plist = []
last_pred = []
for n in range(gs.num_agents):
    plist.append(deque())
    last_pred.append(np.zeros(gs.num_actions))
stochastic = [x for x in range(gs.num_agents)]
while True:
    if model_type in no_epsilon:
        stochastic = []
    got_rewards = []
    did_bad = []
    iteration += 1

    done = 0
    current_iterations += 1
    winner = gs.get_winner()
    if winner is not None:
        done = 1

    if current_iterations >= episode_limit:
        """
        if gs.reinforcements[0] > gs.reinforcements[1]:
            winner = 1
        elif gs.reinforcements[1] > gs.reinforcements[0]:
            winner = 2
        """
        done = 1

    mean_rewards = np.mean(tracked_rewards[-50:])
    team_hp = [[],[]]
    for item in gs.agents:
        x, y, t, u, h = item
        team_hp[t-1].append(h)

    msg = "Model: " + model_type
    msg += " Iter: " + str(iteration)
    msg += " Epis: " + str(episode)
    msg += " Rewards: " + "%.3f"%episode_rewards
    msg += "\nWinner: " + str(winner)
    msg += "\nT1 reinforcements: " + str(gs.reinforcements[0])
    msg += " Commanders: " + str(gs.num_commanders[0])
    msg += " King HP: " + str(gs.kings[0][3])
    #msg += "\nTeam HP: " + ", ".join([str(x) for x in team_hp[0]])
    msg += "\nT2 reinforcements: " + str(gs.reinforcements[1])
    msg += " Commanders: " + str(gs.num_commanders[1])
    msg += " King HP: " + str(gs.kings[1][3])
    #msg += "\nTeam HP: " + ", ".join([str(x) for x in team_hp[1]])

    alive = []
    all_preds = []
    states = []
    states_t1 = []
    actions = []
    rewards = []
    dones = []
    gs.reset_markers()
    for index, agent in enumerate(gs.agents):
        all_preds.append(np.zeros(gs.num_actions))
        alive.append(index)

    for n, index in enumerate(alive):
        x, y, t, u, h = gs.agents[index]
        state = get_state(gs, index, model_type)
        states.append(state)
        pred = models[index].get_action(state)
        if model_type in train_after_episode:
            temp = np.zeros(gs.num_actions, dtype=float)
            temp[pred] = 1.0
            pred = temp
        all_preds[index] = pred
        last_pred[index] = pred
        if index in stochastic:
            move = random.randint(0, gs.num_actions-1)
        else:
            move = np.argmax(pred)
        reward = gs.move_agent(index, move)
        state_t1 = get_state(gs, index, model_type)
        states_t1.append(state_t1)
        if done == 1:
            if winner is not None:
                if t == winner:
                    reward += 1
        episode_rewards += reward
        rewards.append(reward)
        if model_type in sac_models:
            actions.append(pred)
        else:
            actions.append(move)
        dones.append(done)
        gs.update_agent_positions()

    if print_visuals == True:
        os.system('clear')
        print()
        print(gs.print_game_space())
        print(msg)
        if print_preds == True:
            print_all_preds(all_preds)
        time.sleep(0.03)
    else:
        echunk = int(episode_limit*0.05)
        if current_iterations % echunk == 0:
            nchunk = int(current_iterations/echunk)
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write("#"*nchunk)
            sys.stdout.flush()

    if iteration > 0 and iteration % update_iter == 0:
        for n in range(gs.num_agents):
            models[n].update_target()

    for n, a in enumerate(alive):
        models[a].push_replay(states[n], actions[n], rewards[n], done, states_t1[n])

    for n, a in enumerate(alive):
        models[a].train()

    epsilon = max(min_epsilon, 1 - (total_rewards*epsilon_degrade))

    if done == 1:
        if print_visuals == False:
            print()
            print(msg)
        if model_type in train_after_episode:
            for n in range(gs.num_agents):
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write("Training model: " + "%03d"%n)
                sys.stdout.flush()
                models[n].update_policy()
        ns = int(epsilon * gs.num_agents)
        if ns > 0:
            stochastic = [x for x in random.sample(range(gs.num_agents), ns)]
        else:
            stochastic = []
        for n in range(gs.num_agents):
            models[n].save_model(dirname, n)
        filename = os.path.join(dirname, "rewards.json")
        with open(filename, "w") as f:
            f.write(json.dumps(tracked_rewards))
        filename = os.path.join(dirname, "durations.json")
        with open(filename, "w") as f:
            f.write(json.dumps(tracked_durations))
        last_iterations = current_iterations
        tracked_rewards.append(episode_rewards)
        tracked_durations.append(current_iterations)
        last_reward = episode_rewards
        last_winner = winner
        current_iterations = 0
        episode += 1
        episode_rewards = 0
        team_rewards = Counter()
        gs.reset()
