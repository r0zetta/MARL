import random, sys
import numpy as np
from collections import deque

class game_space:
    def __init__(self, width, height, num_agents, walls, bonuses,
                 split_layers=False, flatten_state=False,
                 visible=5, prev_states=4):
        self.split_layers = split_layers
        self.flatten_state = flatten_state
        self.visible = visible
        self.previous_states = prev_states
        self.width = width + (2 * self.visible)
        self.height = height + (2 * self.visible)
        self.num_agents = num_agents
        self.max_bonus = bonuses
        self.agent_hp = 20
        self.sep_left = 0.6
        self.sep_right = 0.6
        self.num_actions = 4 # up, down, left, right
        self.moves = ["l", "r", "u", "d"]
        self.walls = walls
        self.reset()

    def reset(self):
        space = self.make_empty_game_space()
        if self.walls > 0:
            space = self.add_walls(space)
        self.initial_game_space = np.array(space)
        self.game_space = np.array(self.initial_game_space)
        self.create_new_agents()
        self.make_bonuses()
        self.update_agent_positions()

    def create_new_agents(self):
        self.agents = []
        self.agent_states = []
        pad = self.visible
        team1 = random.sample(range(self.num_agents), int(self.num_agents/2))
        player_num = 0
        while len(self.agents) < self.num_agents:
            team = 0
            xpos = random.randint(pad, int(self.width*self.sep_left))
            ypos = random.randint(pad, self.height-(pad*2))
            if player_num in team1:
                team = 1
                xpos = random.randint(int(self.width*self.sep_right), self.width-(pad*2))
                ypos = random.randint(pad, self.height-(pad*2))
            if self.game_space[ypos][xpos] != 0:
                continue
            hp = self.agent_hp
            overlapped = self.check_for_overlap(xpos, ypos)
            if overlapped == False:
                self.agents.append([xpos, ypos, team, hp])
                self.agent_states.append(deque())
                agent_index = len(self.agents)-1
                state_size = self.get_state_size()
                for n in range(self.previous_states):
                    self.agent_states[agent_index].append(np.zeros(state_size, dtype=int))
                if team == 0:
                    self.game_space[ypos][xpos] = 2
                else:
                    self.game_space[ypos][xpos] = 3
                player_num += 1

    def make_empty_game_space(self):
        space = np.zeros((self.height, self.width), dtype=int)
        for n in range(self.width):
            for m in range(self.visible):
                space[m][n] = 1
                space[self.height-(m+1)][n] = 1
        for n in range(self.height):
            for m in range(self.visible):
                space[n][m] = 1
                space[n][self.width-(m+1)] = 1
        return space

    def find_random_empty_cell(self, space):
        xpos = 0
        ypos = 0
        empty = False
        pad = self.visible
        while empty == False:
            xpos = random.randint(pad, self.width-(pad*2))
            ypos = random.randint(pad, self.height-(pad*2))
            if space[ypos][xpos] == 0:
                empty = True
                break
        return xpos, ypos

    def add_walls(self, space):
        added = 0
        target = int((self.width * self.height) * self.walls)
        while added < target:
            xpos, ypos = self.find_random_empty_cell(space)
            space[ypos][xpos] = 1
            for n in range(50):
                move = random.randint(0,3)
                if move == 0:
                    xpos = max(0, xpos-1)
                elif move == 1:
                    xpos = min(self.width-1, xpos+1)
                elif move == 2:
                    ypos = max(0, ypos-1)
                elif move == 3:
                    ypos = min(self.height-1, ypos+1)
                if space[ypos][xpos] == 0:
                    added += 1
                space[ypos][xpos] = 1
                if added >= target:
                    break
        return space

    def make_bonuses(self):
        self.bonuses = []
        while len(self.bonuses) < self.max_bonus:
            x, y = self.find_random_empty_cell(self.game_space)
            self.bonuses.append([x, y])
            self.game_space[y][x] = 4

    def add_bonuses(self, space):
        for item in self.bonuses:
            x, y = item
            space[y][x] = 4
        return space

    def replace_bonus(self, xpos, ypos):
        for index, item in enumerate(self.bonuses):
            x, y = item
            if x == xpos and y == ypos:
                newx, newy = self.find_random_empty_cell(self.game_space)
                self.bonuses[index] = [newx, newy]
                self.game_space[ypos][xpos] = 4

    def add_agents(self, space):
        for item in self.agents:
            x, y, t, h = item
            if h > 0:
                if t == 0:
                    space[y][x] = 2
                elif t == 1:
                    space[y][x] = 3
        return space

    def update_agent_positions(self):
        space = np.array(self.initial_game_space)
        space = self.add_bonuses(space)
        space = self.add_agents(space)
        self.game_space = space

    def get_num_living_agents(self, team_num):
        num_living = 0
        for index, item in enumerate(self.agents):
            x, y, t, h = item
            if t == team_num:
                if h > 0:
                    num_living += 1
        return num_living

    def get_team_hp(self, team_num):
        hps = 0
        for item in self.agents:
            x, y, t, h = item
            if t == team_num:
                hps += h
        return hps

    def get_visible(self, ypos, xpos):
        lpad = self.visible
        rpad = self.visible+1
        left = xpos-lpad
        right = xpos+rpad
        top = ypos-lpad
        bottom = ypos+rpad
        visible = np.array(self.game_space[left:right, top:bottom], dtype=int)
        return visible

    def get_state_size(self):
        state_size = ((self.visible*2)+1)*((self.visible*2)+1)
        if self.split_layers == True:
            return 4 * state_size
        return state_size

    def get_agent_state(self, index):
        agent_states = self.agent_states[index]
        x, y, s, h = self.agents[index]
        visible = self.get_visible(x, y)
        if self.split_layers == True:
            layers = []
            for l in range(1, 5):
                layers.append(np.array((visible == l).astype(int)))
            state = np.ravel(layers)
        else:
            state = np.ravel([visible])
        agent_states.append(state)
        if len(agent_states) < self.previous_states:
            for m in range(self.previous_states - len(agent_states)):
                agent_states.append(state)
        if len(agent_states) > self.previous_states:
            agent_states.popleft()
        states = list(agent_states)
        self.agent_states[index] = deque(states)
        states = np.array(states)
        if self.flatten_state == True:
            return np.ravel(states)
        else:
            return states

    def get_agent_at_position(self, xpos, ypos):
        for index, item in enumerate(self.agents):
            x, y, s, h = item
            if x == xpos and y == ypos:
                return index
        return None

    def hit_agent(self, xpos, ypos):
        index = self.get_agent_at_position(xpos, ypos)
        if index is not None:
            x, y, s, h = self.agents[index]
            if h > 0:
                h = h - 1
                self.agents[index] = [x, y, s, h]
                return index
        return None

    def move_agent(self, index, move):
        reward = 0
        x, y, t, h = self.agents[index]
        newx = x
        newy = y
        if move == 0: # left
            newx = max(0, x-1)
        elif move == 1: # right
            newx = min(self.width-1, x+1)
        elif move == 2: # up
            newy = max(0, y-1)
        elif move == 3: # down
            newy = min(self.height-1, y+1)
        moved = False
        team1 = [2]
        team2 = [3]
        if newx != x or newy != y:
            item = self.game_space[newy][newx]
            if item == 0:
                moved = True
                reward = 0
            elif item == 4:
                moved = True
                reward = 1
                h = self.agent_hp
                self.replace_bonus(newx, newy)
            else:
                if t == 0:
                    if item in team2:
                        if self.hit_agent(newx, newy) is not None:
                            reward = 1
                elif t == 1:
                    if item in team1:
                        if self.hit_agent(newx, newy) is not None:
                            reward = 1

        if moved == True:
            if self.check_for_overlap(newx, newy) == True:
                moved = False

        if moved == True:
            x = newx
            y = newy

        self.agents[index] = [x, y, t, h]
        return reward

    def check_for_overlap(self, xpos, ypos):
        for item in self.agents:
            x, y, s, h = item
            if h > 0 and xpos == x and ypos == y:
                return True
        return False

    def get_printable(self, item):
        if item == 1:
            return "\x1b[1;35;40m" + "▒" + "\x1b[0m"
        elif item == 2:
            return "\x1b[1;32;40m" + "+" + "\x1b[0m"
        elif item == 3:
            return "\x1b[1;33;40m" + "+" + "\x1b[0m"
        elif item == 4:
            return "\x1b[1;31;40m" + "©" + "\x1b[0m"
        else:
            return "\x1b[1;32;40m" + " " + "\x1b[0m"

    def print_game_space(self):
        printable = ""
        pad = self.visible - 1
        for column in self.game_space[pad:-pad]:
            for item in column[pad:-pad]:
                printable += self.get_printable(item)
            printable += "\n"
        return printable

