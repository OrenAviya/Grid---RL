import numpy as np
import random
import math

class GridWorld:
    
    def __init__(self, W, H, L, p, r, gamma=0.5):
        self.W = W
        self.H = H
        self.L = L
        self.p = p
        self.r = r
        self.gamma = gamma
        self.T = 1.0  # Initial temperature for Boltzmann Exploration
        self.q_values = np.zeros((W, H, 4))  # 4 possible actions (up, down, left, right)
        self.policy = np.ones((W, H, 4)) / 4  # Initial policy is uniform random
        self.experience = []
    
    
    def is_terminal(self, state):
        x, y = state
        for (px, py, reward) in self.L:
            if x == px and y == py and reward != 0:
                return True
        return False

    def step(self, state, action):
        x, y = state
        if random.random() > self.p:
            action = (action + random.choice([-1, 1])) % 4

        if action == 0 and y > 0:  # Up
            y -= 1
        elif action == 1 and y < self.H - 1:  # Down
            y += 1
        elif action == 2 and x > 0:  # Left
            x -= 1
        elif action == 3 and x < self.W - 1:  # Right
            x += 1

        # Check if new position is a wall
        if (x, y) in [(px, py) for (px, py, reward) in self.L if reward == 0]:
            x, y = state  # Stay in the same place if it's a wall

        reward = self.r
        for (px, py, rew) in self.L:
            if x == px and y == py:
                reward += rew
                break

        next_state = (x, y)
        return next_state, reward

    def boltzmann_exploration(self, state):
        q_values = self.q_values[state[0], state[1]]
        # Normalize Q-values to prevent overflow
        max_q = np.max(q_values)
        exp_values = np.exp((q_values - max_q) / self.T)
        probabilities = exp_values / np.sum(exp_values)
        action = np.random.choice(np.arange(4), p=probabilities)
        return action

    def update_q_values(self):
        for (i, a, r, j) in self.experience:
            x, y = i
            next_x, next_y = j
            best_next_action = np.argmax(self.q_values[next_x, next_y])
            td_target = r + self.gamma * self.q_values[next_x, next_y, best_next_action]
            self.q_values[x, y, a] += 0.1 * (td_target - self.q_values[x, y, a])

    def update_policy(self):
        for x in range(self.W):
            for y in range(self.H):
                if (x, y) not in [(px, py) for (px, py, reward) in self.L]:
                    best_action = np.argmax(self.q_values[x, y])
                    self.policy[x, y] = np.eye(4)[best_action]

    def run(self, iterations=1000):
        for _ in range(iterations):
            state = (random.randint(0, self.W - 1), random.randint(0, self.H - 1))
            while not self.is_terminal(state):
                action = self.boltzmann_exploration(state)
                next_state, reward = self.step(state, action)
                self.experience.append((state, action, reward, next_state))
                state = next_state
            self.update_q_values()
            self.update_policy()
            self.T = max(0.01, self.T * 0.99)  # Cool down the temperature

    def print_policy(self):
        directions = [" ↑ ", " ↓ ", " ← ", " → ", " █ ", " E "]
        for y in range(self.H):
            for x in range(self.W):
                if (x, y) in [(px, py) for (px, py, reward) in self.L if reward == 0]:
                    best_action = 4
                    print(directions[best_action], end=" ")
                elif (x, y) in [(px, py) for (px, py, reward) in self.L if reward != 0]:
                    best_action = 5
                    print(directions[best_action], end=" ")
                else:
                    best_action = np.argmax(self.policy[x, y])
                    print(directions[best_action], end=" ")
            print()

    def print_q_values(self):
        for y in range(self.H):
            for x in range(self.W):
                if (x, y) not in [(px, py) for (px, py, reward) in self.L]:
                    print(np.max(self.q_values[x, y]), end=" ")
                elif (x, y) in [(px, py) for (px, py, reward) in self.L if reward == 0 ]:
                    print("W" , end=" ")
                elif (x, y) in [(px, py) for (px, py, reward) in self.L if reward != 0 ]:
                    print("E", end=" ")
            print()
class GridWorldQLearning(GridWorld):
    
    def __init__(self, W, H, L, p, r, gamma=0.5):
        super().__init__(W, H, L, p, r, gamma)
    
    def update_q_values(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action = np.argmax(self.q_values[next_x, next_y])
        td_target = reward + self.gamma * self.q_values[next_x, next_y, best_next_action]
        self.q_values[x, y, action] += 0.1 * (td_target - self.q_values[x, y, action])

    def run(self, episodes=1000):
        for _ in range(episodes):
            state = (random.randint(0, self.W - 1), random.randint(0, self.H - 1))
            while not self.is_terminal(state):
                action = self.boltzmann_exploration(state)
                next_state, reward = self.step(state, action)
                self.update_q_values(state, action, reward, next_state)
                state = next_state
            self.update_policy()
            self.T = max(0.01, self.T * 0.99)  # Cool down the temperature
class GridWorldMDP:
    def __init__(self, W, H, L, p, r, gamma=0.5):
        self.W = W
        self.H = H
        self.L = L
        self.p = p
        self.r = r
        self.gamma = gamma
        self.v_values = np.zeros((W, H))  # ערכי V
        self.policy = np.zeros((W, H), dtype=int)  # הפוליסי

    def is_terminal(self, state):
        x, y = state
        for (px, py, reward) in self.L:
            if x == px and y == py and reward != 0:
                return True
        return False

    def step(self, state, action):
        x, y = state
        if action == 0 and y > 0:  # Up
            y -= 1
        elif action == 1 and y < self.H - 1:  # Down
            y += 1
        elif action == 2 and x > 0:  # Left
            x -= 1
        elif action == 3 and x < self.W - 1:  # Right
            x += 1
        
        # Check if new position is a wall
        if (x, y) in [(px, py) for (px, py, reward) in self.L if reward == 0]:
            return state, self.r  # Stay in the same place if it's a wall

        reward = self.r
        for (px, py, rew) in self.L:
            if x == px and y == py:
                reward += rew
                break

        next_state = (x, y)
        return next_state, reward

    def value_iteration(self, iterations=1000):
        for _ in range(iterations):
            new_v_values = np.copy(self.v_values)
            for x in range(self.W):
                for y in range(self.H):
                    if not self.is_terminal((x, y)):
                        action_values = []
                        for action in range(4):
                            next_state, reward = self.step((x, y), action)
                            next_x, next_y = next_state
                            action_value = reward + self.gamma * self.v_values[next_x, next_y]
                            action_values.append(action_value)
                        new_v_values[x, y] = max(action_values)
            self.v_values = new_v_values
        self.update_policy()

    def update_policy(self):
        for x in range(self.W):
            for y in range(self.H):
                if not self.is_terminal((x, y)):
                    action_values = []
                    for action in range(4):
                        next_state, reward = self.step((x, y), action)
                        next_x, next_y = next_state
                        action_value = reward + self.gamma * self.v_values[next_x, next_y]
                        action_values.append(action_value)
                    self.policy[x, y] = np.argmax(action_values)

    def print_policy(self):
        directions = [" ↑ ", " ↓ ", " ← ", " → ", " █ ", " E "]
        for y in range(self.H):
            for x in range(self.W):
                if (x, y) in [(px, py) for (px, py, reward) in self.L if reward == 0]:
                    best_action = 4
                    print(directions[best_action], end=" ")
                elif (x, y) in [(px, py) for (px, py, reward) in self.L if reward != 0]:
                    best_action = 5
                    print(directions[best_action], end=" ")
                else:
                    best_action = self.policy[x, y]
                    print(directions[best_action], end=" ")
            print()

    def print_v_values(self):
        for y in range(self.H):
            for x in range(self.W):
                print( self.v_values[x, y] , end = " ")
            print()

# Example usage:
w = 4
h = 3
L = [(1,1,0),(3,2,1),(3,1,-1)]
p = 0.8
r = 0.04

def transform_coordinates(L, w, h):
        # Assuming L is a list of (x, y, v) tuples
        return [(x, h - 1 - y, v) for x, y, v in L]  # Adjusted for numpy indexing

L = transform_coordinates(L, w, h)

print("transformed L: ", L)

grid_world_mdp = GridWorldMDP(w, h, L, p, r)
grid_world_mdp.value_iteration()
print("policy with MDP: ")
grid_world_mdp.print_policy()
print("values with MDP: ")
grid_world_mdp.print_v_values()

grid_world = GridWorld(w, h, L, p, r)
grid_world.run()
print("policy with mbrl: ")
grid_world.print_policy()
print ("values with mbrl: ")
grid_world.print_q_values()

# Example usage for Q-Learning:
grid_world_qlearning = GridWorldQLearning(w, h, L, p, r)
grid_world_qlearning.run()
print("policy with mfrl: ")
grid_world_qlearning.print_policy()
print("values with mfrl: ")
grid_world_qlearning.print_q_values()