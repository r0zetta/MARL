import random, sys
import numpy as np
from collections import deque

class Agent:
    def __init__(self, index, agent_type, xpos, ypos, state):
        self.index = index
        self.agent_type = agent_type
        self.xpos = xpos
        self.ypos = ypos
        self.last_direction = 0
        self.has_rock = 0
        self.states = state

    def get_value(self):
        # Miner
        if self.agent_type == 0:
            if self.has_rock == 0:
                return 10
            else:
                return 11
        # Collector
        elif self.agent_type == 1:
            if self.has_rock == 0:
                return 20
            else:
                return 21
        # Defender
        else:
            return 30

    def get_position(self):
        return self.xpos, self.ypos, self.has_rock

    def set_position(self, xpos, ypos, last_direction):
        self.xpos = xpos
        self.ypos = ypos
        self.last_direction = last_direction

    def get_rock(self):
        self.has_rock = 1

    def drop_rock(self):
        self.has_rock = 0

class game_space:
    def __init__(self, width, height, num_agents, walls,
                 split_layers=False, flatten_state=False,
                 min_rocks=10, min_holes=4, min_imps=4,
                 visible=5, prev_states=4):
        self.split_layers = split_layers
        self.flatten_state = flatten_state
        self.visible = visible
        self.previous_states = prev_states
        self.width = width + (2 * self.visible)
        self.height = height + (2 * self.visible)
        self.cells = width * height
        self.with_imps = False
        self.num_agents = num_agents
        self.num_initial_rock_piles = max(min_rocks, self.cells * 0.015)
        self.num_holes = max(min_holes, self.cells * 0.006)
        self.num_imps = max(min_imps, self.cells * 0.0001)
        self.num_miners = int(self.num_agents/3)
        self.num_collectors = int(self.num_agents/3)
        self.num_defenders = self.num_agents - self.num_miners - self.num_collectors
        if self.with_imps == False:
            self.num_imps = 0
            self.num_miners = int(self.num_agents/2)
            self.num_collectors = self.num_agents - self.num_miners
            self.num_defenders = 0
        self.num_actions = 5 # up, down, left, right, drop
        self.moves = ["l", "r", "u", "d", "o"]
        self.walls = walls
        self.reset()

    def reset(self):
        space = self.make_empty_game_space()
        if self.walls > 0:
            space = self.add_walls(space)
        self.initial_game_space = np.array(space)
        self.game_space = np.array(self.initial_game_space)
        self.create_holes()
        if self.with_imps:
            self.create_imps()
        self.create_rock_piles()
        self.create_new_agents()
        self.update_agent_positions()

    def create_imps(self):
        self.imps = []
        while len(self.imps) < self.num_imps:
            x, y = self.find_random_empty_cell(self.game_space)
            self.imps.append([x, y])
            self.game_space[y][x] = 5

    def create_holes(self):
        self.holes = []
        while len(self.holes) < self.num_holes:
            x, y = self.find_random_empty_cell(self.game_space)
            self.holes.append([x, y])
            self.game_space[y][x] = 4

    def create_rock_piles(self):
        self.rock_piles = []
        while len(self.rock_piles) < self.num_initial_rock_piles:
            x, y = self.find_random_empty_cell(self.game_space)
            num_rocks = 50
            self.rock_piles.append([x, y, num_rocks])
            self.game_space[y][x] = 3

    def add_holes(self, space):
        for item in self.holes:
            x, y = item
            space[y][x] = 4
        return space

    def add_imps(self, space):
        for item in self.imps:
            x, y = item
            space[y][x] = 5
        return space

    def add_rock_piles(self, space):
        for item in self.rock_piles:
            x, y, n = item
            if n > 0:
                if n > 1:
                    space[y][x] = 3
                else:
                    space[y][x] = 2
        return space

    def add_agents(self, space):
        for agent in self.agents:
            x, y, r = agent.get_position()
            space[y][x] = agent.get_value()
        return space

    def create_new_agents(self):
        self.agents = []
        # Three types of agent:
        # - [0] collectors can only pick up single rocks and deposit them in holes
        # - [1] miners can pick up rocks from rock piles and drop them on empty space
        # - [2] defenders that can eliminate imps
        miners = random.sample(range(self.num_agents), self.num_miners)
        leftovers = list(set(range(self.num_agents)).difference(set(miners)))
        collectors = random.sample(leftovers, self.num_collectors)
        defenders = []
        if self.with_imps == True:
            defenders = list(set(range(self.num_agents)).difference(set(miners).union(set(collectors))))
        pad = self.visible
        agent_index = 0
        state_size = self.get_state_size()
        while agent_index < self.num_agents:
            xpos = random.randint(pad, self.width-(pad*2))
            ypos = random.randint(pad, self.height-(pad*2))
            if self.game_space[ypos][xpos] != 0:
                continue
            overlapped = self.check_for_overlap(xpos, ypos)
            if overlapped == False:
                agent_type = 0
                if agent_index in miners:
                    agent_type = 1
                if agent_index in defenders:
                    agent_type = 2
                states = deque()
                for n in range(self.previous_states):
                    states.append(np.zeros(state_size, dtype=int))
                agent = Agent(agent_index, agent_type, xpos, ypos, states)
                self.agents.append(agent)
                self.game_space[ypos][xpos] = self.agents[agent_index].get_value()
                agent_index += 1

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

    def update_agent_positions(self):
        space = np.array(self.initial_game_space)
        if self.with_imps:
            self.move_imps()
            space = self.add_imps(space)
        space = self.add_holes(space)
        space = self.add_rock_piles(space)
        space = self.add_agents(space)
        self.game_space = space

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
            return 5 * state_size
        return state_size

    def get_agent_state(self, index):
        x, y, r = self.agents[index].get_position()
        visible = self.get_visible(x, y)
        if self.split_layers == True:
            layers = []
            for l in range(1, 6):
                layers.append(np.array((visible == l).astype(int)))
            state = np.ravel(layers)
        else:
            state = np.ravel([visible])
        self.agents[index].states.append(state)
        if len(self.agents[index].states) < self.previous_states:
            for m in range(self.previous_states - len(self.agents[index].states)):
                self.agents[index].states.append(state)
        if len(self.agents[index].states) > self.previous_states:
            self.agents[index].states.popleft()
        states = np.array(list(self.agents[index].states))
        if self.flatten_state == True:
            return np.ravel(states)
        else:
            return states

    def get_rock_pile_at_pos(self, xpos, ypos):
        for index, item in enumerate(self.rock_piles):
            x, y, n = item
            if x == xpos and y == ypos:
                return index

    def take_rock(self, xpos, ypos):
        for index, item in enumerate(self.rock_piles):
            x, y, n = item
            if x == xpos and y == ypos:
                n -= 1
                if n < 1:
                    del(self.rock_piles[index])
                else:
                    self.rock_piles[index] = [x, y, n]

    def drop_rock(self, index):
        # Only miners can drop rocks
        if self.agents[index].agent_type != 1:
            return False
        xpos, ypos, has_rock = self.agents[index].get_position()
        if has_rock == 0:
            return False
        last_direction = self.agents[index].last_direction
        droppedx = xpos
        droppedy = ypos
        if last_direction == 0: #left
            droppedx = droppedx-1
            if droppedx < 1:
                return False
        elif last_direction == 1: #right
            droppedx = droppedx+1
            if droppedx > self.width-1:
                return False
        elif last_direction == 2: #up
            droppedy = droppedy-1
            if droppedy < 1:
                return False
        elif last_direction == 3: #down
            droppedy = droppedy+1
            if droppedy > self.height-1:
                return False
        item = self.game_space[droppedy][droppedx]
        if item == 0: # empty space
            self.rock_piles.append([droppedx, droppedy, 1])
            return True
        return False

    def move_imps(self):
        for index, item in enumerate(self.imps):
            if random.random() < 0.9:
                continue
            x, y = item
            move = random.randint(0, 3)
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
            if self.game_space[newy][newx] == 0:
                moved = True
            # If an imp moves onto a single rock, delete it
            elif self.game_space[newy][newx] == 2:
                rock_pile_index = self.get_rock_pile_at_pos(newx, newy)
                if rock_pile_index is not None:
                    del(self.rock_piles[rock_pile_index])
                moved = True
            if moved == True:
                self.imps[index] = [newx, newy]
        return

    def get_imp_at_pos(self, xpos, ypos):
        for index, item in enumerate(self.imps):
            x, y = item
            if x == xpos and y == ypos:
                return index

    def kill_imp(self, xpos, ypos):
        index = self.get_imp_at_pos(xpos, ypos)
        newx, newy = self.find_random_empty_cell(self.game_space)
        self.imps[index] = [newx, newy]
        self.game_space[newy][newx] = 5

    def move_agent(self, index, move):
        reward = 0
        x, y, r = self.agents[index].get_position()
        last_direction = self.agents[index].last_direction
        agent_type = self.agents[index].agent_type
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
        elif move == 4:
            # Miners can drop the rocks they've mined onto empty cells
            dropped = self.drop_rock(index)
            if dropped == True:
                reward = 1
                self.agents[index].drop_rock()
        moved = False
        if move < 4:
            if self.game_space[newy][newx] == 0:
                moved = True
            else:
                # Collector
                if agent_type == 0:
                    # Pushing into a single rock picks it up (if agent isn't carrying one)
                    if self.game_space[newy][newx] == 2:
                        if r == 0:
                            self.take_rock(newx, newy)
                            self.agents[index].get_rock()
                            reward = 0
                    # Pushing into hole drops rock, if being carried, and gives reward
                    elif self.game_space[newy][newx] == 4:
                        if r > 0:
                            reward = 1
                            self.agents[index].drop_rock()
                # Miner
                elif agent_type == 1:
                    # Pushing into a rock pile mines a rock
                    if self.game_space[newy][newx] == 3:
                        if r == 0:
                            self.take_rock(newx, newy)
                            self.agents[index].get_rock()
                            reward = 0
                # Defender
                else:
                    # If a defender moves into an imp, it is killed and defender gets a reward
                    if self.game_space[newy][newx] == 5:
                        self.kill_imp(newx, newy)
                        reward = 1
                        moved = True

        if moved == True:
            overlap = self.check_for_overlap(newx, newy)
            if overlap != False:
                moved = False

        if moved == True:
            last_direction = move
            x = newx
            y = newy

        self.agents[index].set_position(x, y, last_direction)
        return reward

    def check_for_overlap(self, xpos, ypos):
        for agent in self.agents:
            x, y, s = agent.get_position()
            if xpos == x and ypos == y:
                return agent.index
        return False

    def get_printable(self, item):
        if item == 1:
            return "\x1b[1;37;40m" + "░" + "\x1b[0m"
        elif item == 2:
            return "\x1b[1;35;40m" + "*" + "\x1b[0m"
        elif item == 3:
            return "\x1b[1;34;40m" + "@" + "\x1b[0m"
        elif item == 4:
            return "\x1b[1;36;40m" + "O" + "\x1b[0m"
        elif item == 5:
            return "\x1b[1;31;40m" + "¥" + "\x1b[0m"
        elif item == 10:
            return "\x1b[1;32;40m" + "." + "\x1b[0m"
        elif item == 11:
            return "\x1b[1;32;40m" + "x" + "\x1b[0m"
        elif item == 20:
            return "\x1b[1;33;40m" + "." + "\x1b[0m"
        elif item == 21:
            return "\x1b[1;33;40m" + "x" + "\x1b[0m"
        elif item == 30:
            return "\x1b[1;37;40m" + "x" + "\x1b[0m"
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

