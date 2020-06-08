from rocks_game import *
from misc import *
import time, sys, os
from collections import Counter
import json

game_space_width = 50
game_space_height = 20
preds_len = 20
walls = 0.05
num_agents = 20
min_rocks = 10
min_holes = 4
epsilon_degrade = 0.0001
min_epsilon = 0.0
episode_limit = 4096
max_fill = 0.40
model_type = "PG"
print_visuals = True
print_preds = False

def print_all_preds(all_preds):
    msg = ""
    for index, pred in enumerate(all_preds[:preds_len]):
        x, y, s = gs.agents[index]
        rock = yel + "." + end
        if s > 0:
            rock = gre + "x" + end
        rew = ""
        if index in got_rewards:
            if index in stochastic:
                rew = " +1"
            else:
                rew = gre + " +1" + end
        move = np.argmax(pred)
        plist[index].append(move)
        if len(plist[index]) > 30:
            plist[index].popleft()
        lpred = pretty_vecs(last_pred[index]) + " "
        ppred = pretty_preds(plist[index]) + " "
        ra = gre + str(rpa[index]) + end
        label = gre + "pred" + end
        if index in stochastic:
            ra = str(rpa[index])
            label = "stoc"
        msg += label + "("+"%02d"%index+")" + rock + ":" + ppred + lpred + ra + rew + "\n" 
    print(msg)

dirname = model_type + "_rocks_game_save"
if not os.path.exists(dirname):
    os.makedirs(dirname)
#random.seed(1)
flatten_state, prev_states = get_state_params(model_type)
gs = game_space(game_space_width, game_space_height, num_agents, walls,
                min_rocks=min_rocks, min_holes=min_holes,
                visible=visible, prev_states=prev_states,
                flatten_state=flatten_state, split_layers=False)

models = []
for n in range(gs.num_agents):
    model = get_model(model_type, gs)
    if model.load_model(dirname, n) == True:
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
last_reward = 0
current_iterations = 0
last_iterations = 0
iteration = 0
rpa = Counter()
plist = []
last_pred = []
got_rewards = []
for n in range(gs.num_agents):
    plist.append(deque())
    last_pred.append(np.zeros(gs.num_actions))
stochastic = [x for x in range(gs.num_agents)]
while True:
    if model_type in no_epsilon:
        stochastic = []
    got_rewards = []
    iteration += 1
    done = 0
    current_iterations += 1
    num_rock_piles = len(gs.rock_piles)
    game_space_size = game_space_width * game_space_height
    percent_filled = num_rock_piles/game_space_size
    if percent_filled > max_fill or num_rock_piles < 1 or current_iterations >= episode_limit:
        done = 1

    mean_rewards = np.mean(tracked_rewards[-50:])

    msg = "Model: " + model_type
    msg += " Iter: " + str(iteration)
    msg += " Epis: " + str(episode)
    msg += " Epsi: " + "%.4f"%epsilon
    msg += " Stoc: " + str(len(stochastic))
    msg += " Filled percent: " + "%.2f"%(percent_filled*100)
    msg += "\nCurrent iter: " + str(current_iterations)
    msg += " Last iter: " + str(last_iterations)
    msg += "\nRewards: " + str(episode_rewards)
    msg += " Last: " + str(last_reward)
    msg += " Total: " + str(total_rewards)
    msg += " Mean: " + "%.2f"%mean_rewards

    states = []
    states_t1 = []
    actions = []
    rewards = []
    dones = []
    all_preds = []

    for index, agent in enumerate(gs.agents):
        state = get_state(gs, index, model_type)
        states.append(state)
        pred = models[index].get_action(state)
        if model_type in train_after_episode:
            temp = np.zeros(gs.num_actions, dtype=float)
            temp[pred] = 1.0
            pred = temp
        all_preds.append(pred)
        last_pred[index] = pred
        if index in stochastic:
            move = random.randint(0, gs.num_actions-1)
        else:
            move = np.argmax(pred)
        reward  = gs.move_agent(index, move)
        state_t1 = get_state(gs, index, model_type)
        states_t1.append(state_t1)
        if reward == 1:
            got_rewards.append(index)
            rpa[index] += reward
            episode_rewards += reward
            total_rewards += reward
        rewards.append(reward)
        dones.append(done)
        if model_type in sac_models:
            actions.append(pred)
        else:
            actions.append(move)
        gs.update_agent_positions()

    if print_visuals == True:
        os.system('clear')
        print()
        print(gs.print_game_space())
        print(msg)
        if print_preds == True:
            print_all_preds(all_preds)
        time.sleep(0.05)
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

    for n in range(gs.num_agents):
        models[n].push_replay(states[n], actions[n], rewards[n], dones[n], states_t1[n])

    for n in range(gs.num_agents):
        models[n].train()

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
        print("Saving...")
        for n in range(gs.num_agents):
            models[n].save_model(dirname, n)
        filename = os.path.join(dirname, "rewards.json")
        with open(filename, "w") as f:
            f.write(json.dumps(tracked_rewards))
        filename = os.path.join(dirname, "durations.json")
        with open(filename, "w") as f:
            f.write(json.dumps(tracked_durations))
        print("Done saving...")
        episode += 1
        tracked_rewards.append(episode_rewards)
        tracked_durations.append(current_iterations)
        last_reward = episode_rewards
        last_iterations = current_iterations
        current_iterations = 0
        episode_rewards = 0
        gs.reset()
