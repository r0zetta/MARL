import random, sys
import numpy as np
from collections import deque

class game_space:
    def __init__(self, split_layers=False, flatten_state=False,
                 visible=5, prev_states=4):
        self.split_layers = split_layers
        self.flatten_state = flatten_state
        self.visible = visible
        self.agent_hp = 10
        self.commander_hp = 100
        self.lieutenant_hp = 25
        self.lieutenant_damage = 2
        self.small_reward = 0.001
        self.medium_reward = 0.01
        self.num_reinforcements = 300
        self.got_hit = []
        self.got_healed = []
        self.previous_states = prev_states
        self.initial_area = self.get_game_area()
        self.height, self.width = self.initial_area.shape
        self.num_actions = 5 # up, down, left, right, action
        self.moves = ["l", "r", "u", "d", "a"]
        self.reset()

    def get_game_area(self):
        area = []
        with open("av_game_area.txt", "r") as f:
            for line in f:
                area.append([int(x) for x in line.strip()])
        temp = np.array(area)
        height, width = temp.shape
        new_area = []
        new_height = height + 2 * (self.visible-1)
        new_width = width + 2 * (self.visible-1)
        for n in range(self.visible-1):
            new_area.append([1]*new_width)
        for row in area:
            new_row = [1]*(self.visible-1) + row + [1]*(self.visible-1)
            new_area.append(new_row)
        for n in range(self.visible-1):
            new_area.append([1]*new_width)
        return np.array(new_area)

    def reset(self):
        self.initial_game_space = np.array(np.where(self.initial_area > 1, 0, self.initial_area))
        self.game_space = np.array(self.initial_game_space)
        self.reinforcements = [self.num_reinforcements, self.num_reinforcements]
        self.create_new_agents()
        self.create_lieutenants()
        self.create_commanders()
        self.update_agent_positions()

    def reset_markers(self):
        self.got_hit = []
        self.got_healed = []

    def create_lieutenants(self):
        self.lieutenants = []
        self.num_lieutenants = [0,0]
        hp = self.lieutenant_hp
        for item in np.argwhere(self.initial_area==2):
            ypos, xpos = item
            team = 1
            self.lieutenants.append([xpos, ypos, team, hp])
            self.game_space[ypos, xpos] = 0
            self.num_lieutenants[0] += 1
        for item in np.argwhere(self.initial_area==3):
            ypos, xpos = item
            team = 2
            self.lieutenants.append([xpos, ypos, team, hp])
            self.game_space[ypos, xpos] = 0
            self.num_lieutenants[1] += 1

    def create_commanders(self):
        self.commanders = []
        hp = self.commander_hp
        for item in np.argwhere(self.initial_area==4):
            ypos, xpos = item
            team = 1
            self.commanders.append([xpos, ypos, team, hp])
            self.game_space[ypos, xpos] = 0
        for item in np.argwhere(self.initial_area==5):
            ypos, xpos = item
            team = 2
            self.commanders.append([xpos, ypos, team, hp])
            self.game_space[ypos, xpos] = 0

    def create_new_agents(self):
        t1_start_pos = []
        for item in np.argwhere(self.initial_area==6):
            ypos, xpos = item
            xpos = xpos - self.visible
            ypos = ypos - self.visible
            t1_start_pos.append([ypos, xpos])
            self.game_space[ypos, xpos] = 0
        t2_start_pos = []
        for item in np.argwhere(self.initial_area==7):
            ypos, xpos = item
            xpos = xpos - self.visible
            ypos = ypos - self.visible
            t2_start_pos.append([ypos, xpos])
            self.game_space[ypos, xpos] = 0
        self.agents = []
        self.agent_states = []
        pad = self.visible
        state_size = self.get_state_size()
        self.spawn_points = []
        for team in [1, 2]:
            for index, unit_type in enumerate([0, 0, 0, 0, 0, 1, 1, 1, 2, 2]):
                x, y = 0, 0
                if team == 1:
                    y, x = t1_start_pos[index]
                else:
                    y, x = t2_start_pos[index]
                xpos = x+self.visible
                ypos = y+self.visible
                hp = self.agent_hp
                self.agents.append([xpos, ypos, team, unit_type, hp])
                self.agent_states.append(deque())
                agent_index = len(self.agents)-1
                state_size = self.get_state_size()
                for n in range(self.previous_states):
                    self.agent_states[agent_index].append(np.zeros(state_size, dtype=int))
                self.game_space[ypos][xpos] = 10*team + unit_type
                self.spawn_points.append([xpos, ypos])
        self.num_agents = len(self.agents)

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

    def add_agents(self, space):
        for item in self.agents:
            xpos, ypos, t, u, h = item
            if h > 0:
                space[ypos][xpos] = 10*t + u
            else:
                space[ypos][xpos] = 0
        return space

    def add_lieutenants(self, space):
        for item in self.lieutenants:
            xpos, ypos, t, h = item
            if h > 0:
                space[ypos][xpos] = 1+t
            else:
                space[ypos][xpos] = 0
        return space

    def add_commanders(self, space):
        for item in self.commanders:
            xpos, ypos, t, h = item
            space[ypos][xpos] = 3+t
        return space

    def get_lieutenant_at_position(self, xpos, ypos):
        for index, item in enumerate(self.lieutenants):
            x, y, t, h = item
            if x == xpos and y == ypos:
                return index

    def get_commander_at_position(self, xpos, ypos):
        for index, item in enumerate(self.commanders):
            x, y, t, h = item
            if x == xpos and y == ypos:
                return index

    def hit_lieutenant(self, xpos, ypos, dmg):
        index = self.get_lieutenant_at_position(xpos, ypos)
        x, y, t, h = self.lieutenants[index]
        self.got_hit.append([x, y])
        h = max(0, h-dmg)
        self.lieutenants[index] = [x, y, t, h]
        lieutenants_alive = [0,0]
        for item in self.lieutenants:
            x, y, t, h = item
            if t == 1:
                if h > 0:
                    lieutenants_alive[0] += 1
            elif t == 2:
                if h > 0:
                    lieutenants_alive[1] += 1
        self.num_lieutenants = list(lieutenants_alive)

    def hit_commander(self, xpos, ypos, dmg):
        index = self.get_commander_at_position(xpos, ypos)
        x, y, t, h = self.commanders[index]
        self.got_hit.append([x, y])
        h = max(0, h-dmg)
        self.commanders[index] = [x, y, t, h]
        return h

    def update_agent_positions(self):
        space = np.array(self.initial_game_space)
        space = self.add_lieutenants(space)
        space = self.add_commanders(space)
        space = self.add_agents(space)
        self.game_space = space

    def get_visible(self, ypos, xpos, visible):
        lpad = visible
        rpad = visible+1
        left = xpos-lpad
        right = xpos+rpad
        top = ypos-lpad
        bottom = ypos+rpad
        visible = np.array(self.game_space[left:right, top:bottom], dtype=int)
        return visible

    def get_item_in_visible(self, xpos, ypos, item, visible):
        area = self.get_visible(xpos, ypos, visible)
        coords = np.argwhere(area==item)
        new_coords = []
        for c in coords:
            y, x = c
            y = ypos-visible+y
            x = xpos-visible+x
            new_coords.append([x, y])
        return new_coords

    def get_mage_target(self, xpos, ypos, targets):
        enemy_coords = []
        for t in targets:
            coords = self.get_item_in_visible(xpos, ypos, t, 2)
            if len(coords) > 0:
                for c in coords:
                    enemy_coords.append(c)
        if len(enemy_coords) > 0:
            target = random.choice(enemy_coords)
            return target
        return False

    def mage_shoot(self, index):
        xpos, ypos, team, u, h = self.agents[index]
        targets = []
        # Check for enemy team
        if team == 1:
            targets = [20, 21, 22]
        else:
            targets = [10, 11, 12]
        coords = self.get_mage_target(xpos, ypos, targets)
        if coords is not False:
            x, y = coords
            index = self.get_agent_at_position(x, y)
            x, y, t, u, h = self.agents[index]
            if u % 10 == 0:
                self.hit_agent(x, y, 1)
            else:
                self.hit_agent(x, y, 2)
            return 1
        # Check for enemy commander
        if team == 1:
            targets = [5]
        else:
            targets = [4]
        coords = self.get_mage_target(xpos, ypos, targets)
        if coords is not False:
            x, y = coords
            self.hit_commander(x, y, 2)
            commander_damage = self.get_commander_damage(x, y)
            self.hit_agent(xpos, ypos, commander_damage)
            return 10
        # Check for enemy lieutenants
        if team == 1:
            targets = [3]
        else:
            targets = [2]
        coords = self.get_mage_target(xpos, ypos, targets)
        if coords is not False:
            x, y = coords
            self.hit_lieutenant(x, y, 2)
            self.hit_agent(xpos, ypos, self.lieutenant_damage)
            return 5
        return False

    def heal(self, index):
        xpos, ypos, team, u, h = self.agents[index]
        healed = False
        targets = []
        if team == 1:
            targets = [10, 11, 12]
        else:
            targets = [20, 21, 22]
        friend_coords = []
        for t in targets:
            coords = self.get_item_in_visible(xpos, ypos, t, 3)
            if len(coords) > 0:
                for c in coords:
                    friend_coords.append(c)
        if len(friend_coords) > 0:
            indices = []
            hitpoints = []
            for item in friend_coords:
                x, y = item
                index = self.get_agent_at_position(x, y)
                if index is not None:
                    indices.append(index)
                    x, y, t, u, h = self.agents[index]
                    hitpoints.append(h)
            if len(indices) > 0:
                lowest_health = np.min(hitpoints)
                lowest_health_index = indices[np.argmin(hitpoints)]
                x, y, t, u, h = self.agents[lowest_health_index]
                newh = min(self.agent_hp, h + 2)
                if newh != h:
                    healed = True
                    self.got_healed.append([x, y])
                self.agents[lowest_health_index] = [x, y, t, u, newh]
        return healed

    def get_state_size(self):
        state_size = ((self.visible*2)+1)*((self.visible*2)+1)
        if self.split_layers == True:
            return 4 * state_size
        return state_size

    def get_agent_state(self, index):
        agent_states = self.agent_states[index]
        x, y, t, u, h = self.agents[index]
        visible = self.get_visible(x, y, self.visible)
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
            x, y, t, u, h = item
            if x == xpos and y == ypos:
                return index
        return None

    def hit_agent(self, xpos, ypos, dmg):
        index = self.get_agent_at_position(xpos, ypos)
        x, y, t, u, h = self.agents[index]
        self.got_hit.append([x, y])
        h = max(0, h-dmg)
        if h > 0:
            self.agents[index] = [x, y, t, u, h]
        else: # agent died - respawn and decrement reinforcements counter
            newx, newy = self.spawn_points[index]
            self.agents[index] = [newx, newy, t, u, self.agent_hp]
            self.reinforcements[t-1] = self.reinforcements[t-1] - 1
        return index

    def get_winner(self):
        winner = None
        commander_hp = {}
        for item in self.commanders:
            x, y, t, h = item
            commander_hp[t] = h
        if self.reinforcements[0] < 1:
            return 2
        elif self.reinforcements[1] < 1:
            return 1
        elif commander_hp[1] < 1:
            return 2
        elif commander_hp[2] < 1:
            return 1
        else:
            return None

    def get_commander_damage(self, xpos, ypos):
        index = self.get_commander_at_position(xpos, ypos)
        x, y, t, h = self.commanders[index]
        damage = 2 * self.num_lieutenants[t-1]
        return damage

    def move_agent(self, index, move):
        reward = 0
        x, y, t, u, h = self.agents[index]
        newx = x
        newy = y
        if move == 0: # left
            newx = x-1
        elif move == 1: # right
            newx = x+1
        elif move == 2: # up
            newy = y-1
        elif move == 3: # down
            newy = y+1
        moved = False
        item = self.game_space[newy][newx]
        if item == 0:
            moved = True
        else:
            if t == 1:
                dmg = 1
                if item in [20, 21, 22]:
                    if item != 20 and u == 0:
                        dmg = 2
                    self.hit_agent(newx, newy, dmg)
                    reward = self.small_reward
                elif item == 3:
                    if u == 0:
                        dmg = 2
                    self.hit_lieutenant(newx, newy, dmg)
                    self.hit_agent(x, y, self.lieutenant_damage)
                    x, y, t, u, h = self.agents[index]
                    reward = self.small_reward * 5
                elif item == 5:
                    if u == 0:
                        dmg = 2
                    self.hit_commander(newx, newy, dmg)
                    commander_damage = self.get_commander_damage(newx, newy)
                    self.hit_agent(x, y, commander_damage)
                    x, y, t, u, h = self.agents[index]
                    reward = self.small_reward * 10
            elif t == 2:
                dmg = 1
                if item in [10, 11, 12]:
                    if item != 10 and u == 0:
                        dmg = 2
                    self.hit_agent(newx, newy, dmg)
                    reward = self.small_reward
                elif item == 2:
                    if u == 0:
                        dmg = 2
                    self.hit_lieutenant(newx, newy, dmg)
                    self.hit_agent(x, y, self.lieutenant_damage)
                    x, y, t, u, h = self.agents[index]
                    reward = self.small_reward * 5
                elif item == 4:
                    if u == 0:
                        dmg = 2
                    self.hit_commander(newx, newy, dmg)
                    commander_damage = self.get_commander_damage(newx, newy)
                    self.hit_agent(x, y, commander_damage)
                    x, y, t, u, h = self.agents[index]
                    reward = self.small_reward * 10
        if move == 4:
            if t == 1:
                if u == 0: # t1 soldier
                    pass
                elif u == 1: # t1 mage
                    ret = self.mage_shoot(index)
                    if ret != False:
                        reward = self.small_reward * ret
                        x, y, t, u, h = self.agents[index]
                elif u == 2: # t1 healer
                    ret = self.heal(index)
                    if ret == True:
                        reward = self.small_reward * 10
                        x, y, t, u, h = self.agents[index]
            elif t == 2:
                if u == 0: # t2 soldier
                    pass
                elif u == 1: # t2 mage
                    ret = self.mage_shoot(index)
                    if ret != False:
                        reward = self.small_reward * ret
                        x, y, t, u, h = self.agents[index]
                elif u == 2: # t2 healer
                    ret = self.heal(index)
                    if ret == True:
                        reward = self.small_reward * 10
                        x, y, t, u, h = self.agents[index]

        if moved == True:
            if self.check_for_overlap(newx, newy) == True:
                moved = False

        if moved == True:
            reward = self.small_reward * 0.01
            x = newx
            y = newy

        self.agents[index] = [x, y, t, u, h]
        return reward

    def check_for_overlap(self, xpos, ypos):
        for item in self.agents:
            x, y, t, u, h = item
            if h > 0 and xpos == x and ypos == y:
                return True
        return False

    def was_hit(self, xpos, ypos):
        for item in self.got_hit:
            x, y = item
            if x == xpos and y == ypos:
                return True
        return False

    def was_healed(self, xpos, ypos):
        for item in self.got_healed:
            x, y = item
            if x == xpos and y == ypos:
                return True
        return False

    def get_printable(self, item, hit, healed):
        bg = "40"
        if hit == True:
            bg = "41"
        if healed == True:
            bg = "42"
        if item == 1: # wall
            return "\x1b[2;37;40m" + "▒" + "\x1b[0m"
        elif item == 2: # t1 lieutenant
            return "\x1b[2;35;" + str(bg) + "m" + "o" + "\x1b[0m"
        elif item == 3: # t2 lieutenant
            return "\x1b[2;36;" + str(bg) + "m" + "o" + "\x1b[0m"
        elif item == 4: # t1 commander
            return "\x1b[1;33;" + str(bg) + "m" + "@" + "\x1b[0m"
        elif item == 5: # t2 commander
            return "\x1b[1;32;" + str(bg) + "m" + "@" + "\x1b[0m"
        elif item == 10: # t1 soldier
            return "\x1b[1;33;" + str(bg) + "m" + "+" + "\x1b[0m"
        elif item == 11: # t1 mage
            return "\x1b[1;33;" + str(bg) + "m" + "x" + "\x1b[0m"
        elif item == 12: # t1 healer
            return "\x1b[1;33;" + str(bg) + "m" + "#" + "\x1b[0m"
        elif item == 20: # t2 soldier
            return "\x1b[1;32;" + str(bg) + "m" + "+" + "\x1b[0m"
        elif item == 21: # t2 mage
            return "\x1b[1;32;" + str(bg) + "m" + "x" + "\x1b[0m"
        elif item == 22: # t2 healer
            return "\x1b[1;32;" + str(bg) + "m" + "#" + "\x1b[0m"
        else:
            return "\x1b[1;32;40m" + " " + "\x1b[0m"

    def print_game_space(self):
        printable = ""
        pad = self.visible - 1
        for y, column in enumerate(self.game_space[pad:-pad]):
            ypos = y + pad
            for x, item in enumerate(column[pad:-pad]):
                xpos = x + pad
                hit = self.was_hit(xpos, ypos)
                healed = self.was_healed(xpos, ypos)
                printable += self.get_printable(item, hit, healed)
            printable += "\n"
        return printable

