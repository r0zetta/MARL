from tag_game import *
from misc import *
import time, sys, os, re
from collections import Counter

game_space_width = 60
game_space_height = 25
preds_len = 20
num_agents = 20
walls = 0.03
bonuses = 1
epsilon_degrade = 0.00003
min_epsilon = 0.0
episode_limit = 512
model_type = "PG"
print_preds = False
print_visuals = True

def print_all_preds(all_preds):
    msg = ""
    for index, pred in enumerate(all_preds[:preds_len]):
        x, y, s = gs.agents[index]
        cap = ""
        first = ""
        rew = ""
        if index in got_bonus:
            rew = red + " +1" + end
        if s == 0:
            first = gre + "pred" + end
        else:
            first = yel + "pred" + end
        if index in captured:
            if s == 0:
                cap = " " + gre + "+1" + end
            else:
                cap = " " + yel + "+1" + end
        if index in got_captured:
            cap = " " + red + "-1" + end
        move = np.argmax(pred)
        plist[index].append(move)
        if len(plist[index]) > 30:
            plist[index].popleft()
        lpred = pretty_vecs(last_pred[index]) + " "
        ppred = pretty_preds(plist[index]) + " "
        msg += first + "("+"%02d"%index+"):" + ppred + lpred + cap + rew + "\n" 
    print(msg)

dirname = model_type + "_tag_game_save"

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
gs = game_space(game_space_width, game_space_height, num_agents, walls, bonuses,
                visible=visible, prev_states=prev_states,
                flatten_state=flatten_state)

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

last_episode_length = 0
current_iterations = 0
episode_rewards = 0
previous_winner = 0
team_bonuses = Counter()
team_rewards = Counter()
iteration = 0
plist = []
last_pred = []
got_bonus = []
captured = []
got_captured = []
for n in range(num_agents):
    plist.append(deque())
    last_pred.append(np.zeros(gs.num_actions, dtype=int))
stochastic = [x for x in range(gs.num_agents)]
while True:
    if model_type in no_epsilon:
        stochastic = []
    got_bonus = []
    captured = []
    got_captured = []
    iteration += 1
    team1, team2 = gs.get_team_stats()
    current_iterations += 1
    done = 0
    winning = 0
    if team_bonuses[1] > team_bonuses[0]:
        winning = 1
    if team1 < 3 or team2 < 3 or current_iterations >= episode_limit:
        done = 1

    msg = "Model: " + model_type
    msg += " Iteration: " + str(iteration)
    msg += " Episode: " + str(episode)
    msg += " Epsilon: " + "%.4f"%epsilon
    msg += "\nTeam 1 size: " + str(team1) 
    msg += " T1 score: " + str(team_bonuses[0])
    msg += " T1 rewards: " + str(team_rewards[0])
    msg += "\nTeam 2 size: " + str(team2) 
    msg += " T2 score: " + str(team_bonuses[1])
    msg += " T2 rewards: " + str(team_rewards[1])
    msg += "\nLast winner: " + str(previous_winner+1)
    msg += " Winning: " + str(winning+1)
    msg += "\nEpisode length: " + str(current_iterations)
    msg += " Last length: " + str(last_episode_length)
    msg += " Total rewards: " + "%.2f"%total_rewards

    states = []
    states_t1 = []
    actions = []
    rewards = []
    dones = np.zeros(num_agents, dtype=int)
    all_preds = []
    for index, agent in enumerate(gs.agents):
        x, y, s = gs.agents[index]
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
        total_rewards += reward
        team_rewards[s] += reward
        if reward == 1:
            episode_rewards += reward
            team_bonuses[s] += 1
            got_bonus.append(index)
        rewards.append(reward)
        if model_type in sac_models:
            actions.append(pred)
        else:
            actions.append(move)
        gs.update_agent_positions()

    rewards2 = gs.change_team_status()
    for i, r in enumerate(rewards2):
        x, y, s = gs.agents[i]
        if r == 1:
            captured.append(i)
            total_rewards += 1
        if r == -1:
            got_captured.append(i)
        rewards[i] += r
        team_rewards[s] += r

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
        models[n].push_replay(states[n], actions[n], rewards[n], done, states_t1[n])

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
        for n in range(gs.num_agents):
            models[n].save_model(dirname, n)
        filename = os.path.join(dirname, "rewards.json")
        with open(filename, "w") as f:
            f.write(json.dumps(tracked_rewards))
        filename = os.path.join(dirname, "durations.json")
        with open(filename, "w") as f:
            f.write(json.dumps(tracked_durations))
        episode += 1
        previous_winner = winning
        tracked_rewards.append(episode_rewards)
        tracked_durations.append(current_iterations)
        team_bonuses = Counter()
        team_rewards = Counter()
        last_episode_length = current_iterations
        current_iterations = 0
        episode_rewards = 0
        gs.reset()

