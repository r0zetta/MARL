import random, sys
import numpy as np
from collections import deque

class game_space:
    def __init__(self, width, height, num_agents, walls,
                 split_layers=False, flatten_state=False,
                 min_rocks=10, min_holes=4,
                 visible=5, prev_states=4):
        self.split_layers = split_layers
        self.flatten_state = flatten_state
        self.visible = visible
        self.previous_states = prev_states
        self.width = width + (2 * self.visible)
        self.height = height + (2 * self.visible)
        self.cells = width * height
        self.num_agents = num_agents
        self.num_initial_rock_piles = max(min_rocks, self.cells * 0.015)
        self.num_holes = max(min_holes, self.cells * 0.006)
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
        self.create_rock_piles()
        self.create_new_agents()
        self.update_agent_positions()

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
            self.game_space[y][x] = 5

    def add_holes(self, space):
        for item in self.holes:
            x, y = item
            space[y][x] = 4
        return space

    def add_rock_piles(self, space):
        for item in self.rock_piles:
            x, y, n = item
            if n > 0:
                space[y][x] = 5
        return space

    def add_agents(self, space):
        for item in self.agents:
            x, y, r = item
            if r == 0:
                space[y][x] = 2
            if r == 1:
                space[y][x] = 3
        return space

    def create_new_agents(self):
        self.agents = []
        self.agent_states = []
        pad = self.visible
        while len(self.agents) < self.num_agents:
            xpos = random.randint(pad, self.width-(pad*2))
            ypos = random.randint(pad, self.height-(pad*2))
            if self.game_space[ypos][xpos] != 0:
                continue
            has_rock = 0
            overlapped = self.check_for_overlap(xpos, ypos)
            if overlapped == False:
                self.agents.append([xpos, ypos, has_rock])
                self.agent_states.append(deque())
                agent_index = len(self.agents)-1
                state_size = self.get_state_size()
                for n in range(self.previous_states):
                    self.agent_states[agent_index].append(np.zeros(state_size, dtype=int))
                self.game_space[ypos][xpos] = 2

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
        agent_states = self.agent_states[index]
        x, y, s = self.agents[index]
        visible = self.get_visible(x, y)
        if self.split_layers == True:
            layers = []
            for l in range(1, 6):
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

    def take_rock(self, xpos, ypos):
        for index, item in enumerate(self.rock_piles):
            x, y, n = item
            if x == xpos and y == ypos:
                n -= 1
                if n < 1:
                    del(self.rock_piles[index])
                else:
                    self.rock_piles[index] = [x, y, n]

    def drop_rock(self, xpos, ypos):
        found_index = None
        for index, item in enumerate(self.rock_piles):
            x, y, n = item
            if x == xpos and y == ypos:
                found_index = index
        if found_index is not None:
            x, y, n = self.rock_piles[found_index]
            self.rock_piles[found_index] = [x, y, n+1]
        else:
            self.rock_piles.append([xpos, ypos, 1])

    def move_agent(self, index, move):
        reward = 0
        x, y, r = self.agents[index]
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
            if r > 0: # Drop a rock to the right
                # If a rock is dropped into the deposit zone, get a reward, and the rock is gone
                if self.game_space[y][x+1] == 4:
                    reward = 1
                    r = 0
                # Drop a rock on the ground to the right, if nothing else is there
                elif self.game_space[y][x+1] == 0:
                    reward = 0
                    self.drop_rock(x+1, y)
                    r = 0
        moved = False
        if move < 4:
            # Pushing into a rock picks it up (if agent isn't carrying one)
            if self.game_space[newy][newx] == 5:
                if r == 0:
                    r = 1
                    self.take_rock(newx, newy)
                    reward = 0
            # Pushing into deposit zone drops rock, if being carried, and gives reward
            elif self.game_space[newy][newx] == 4:
                if r > 0:
                    r = 0
                    reward = 1
            elif self.game_space[newy][newx] == 0:
                moved = True

        if moved == True:
            overlap = self.check_for_overlap(newx, newy)
            if overlap != False:
                moved = False
            # Pushing into another agent
            # If A doesn't have a rock and B does, A takes it
            # If A has a rock and B doesn't, B takes it
            if overlap != False:
                xx, yy, rr = self.agents[overlap]
                if r == 1 and rr == 0:
                    self.agents[overlap] = [xx, yy, 1]
                    r = 0
                elif r == 0 and rr == 1:
                    self.agents[overlap] = [xx, yy, 0]
                    r = 1
                moved = False

        if moved == True:
            reward = 0
            x = newx
            y = newy

        self.agents[index] = [x, y, r]
        return reward

    def check_for_overlap(self, xpos, ypos):
        for index, item in enumerate(self.agents):
            x, y, s = item
            if xpos == x and ypos == y:
                return index
        return False

    def get_printable(self, item):
        if item == 1:
            return "\x1b[1;37;40m" + "░" + "\x1b[0m"
        elif item == 2:
            return "\x1b[1;32;40m" + "." + "\x1b[0m"
        elif item == 3:
            return "\x1b[1;32;40m" + "x" + "\x1b[0m"
        elif item == 4:
            return "\x1b[1;33;40m" + "O" + "\x1b[0m"
        elif item == 5:
            return "\x1b[1;35;40m" + "■" + "\x1b[0m"
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

