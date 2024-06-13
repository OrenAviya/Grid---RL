import numpy as np 
import random

# # נתונים לדוגמה
# W, H = 5, 5  # מימדי הלוח
# L = [(1, 2, 10), (3, 4, -5), (2, 2, 0)]  # פרסים, עונשים וקירות
# p = 0.8  # סיכוי הצלחת הצעד
# r = -1  # עלות הצעד
gamma = 0.5  # גורם הנחה

# # יצירת לוח המשחק
# grid = np.zeros((W, H))
# for (x, y, reward) in L:
#     grid[x, y] = reward

W, H = 6, 6
L = [(1, 1, 10), (4, 4, -10), (2, 2, 5), (3, 3, -5)]
p = 0.8
r = -1

# W, H = 4, 4
# L = [(0, 3, 10), (3, 0, -10)]
# p = 0.9
# r = -1

grid = np.zeros((W, H))
for (x, y, reward) in L:
    grid[x, y] = reward


def value_iteration(grid, p, r, gamma, theta=0.01):
    W, H = grid.shape
    V = np.zeros((W, H))
    policy = np.zeros((W, H), dtype=str)
    actions = ['up', 'down', 'left', 'right']
    delta = float('inf')
    
    while delta > theta:
        delta = 0
        for x in range(W):
            for y in range(H):
                v = V[x, y]
                max_value = float('-inf')
                best_action = None
                for action in actions:
                    new_x, new_y = x, y
                    if action == 'up':
                        new_x = max(x - 1, 0)
                    elif action == 'down':
                        new_x = min(x + 1, W - 1)
                    elif action == 'left':
                        new_y = max(y - 1, 0)
                    elif action == 'right':
                        new_y = min(y + 1, H - 1)
                    
                    reward = grid[new_x, new_y]
                    value = p * (reward + gamma * V[new_x, new_y]) + (1 - p) * (reward + gamma * V[x, y]) - r
                    if value > max_value:
                        max_value = value
                        best_action = action
                
                V[x, y] = max_value
                policy[x, y] = best_action
                delta = max(delta, abs(v - V[x, y]))
    
    return V, policy


def bellman_equation(grid, p, r, gamma, theta=0.01):
    W, H = grid.shape
    V = np.zeros((W, H))
    delta = float('inf')
    
    while delta > theta:
        delta = 0
        for x in range(W):
            for y in range(H):
                v = V[x, y]
                max_value = float('-inf')
                for action in ['up', 'down', 'left', 'right']:
                    new_x, new_y = x, y
                    if action == 'up':
                        new_x = max(x - 1, 0)
                    elif action == 'down':
                        new_x = min(x + 1, W - 1)
                    elif action == 'left':
                        new_y = max(y - 1, 0)
                    elif action == 'right':
                        new_y = min(y + 1, H - 1)
                    
                    reward = grid[new_x, new_y]
                    value = p * (reward + gamma * V[new_x, new_y]) + (1 - p) * (reward + gamma * V[x, y]) - r
                    max_value = max(max_value, value)
                
                V[x, y] = max_value
                delta = max(delta, abs(v - V[x, y]))
    
    return V
    
def q_learning(grid, p, r, gamma, alpha=0.1, epsilon=0.1, episodes=1000):
    W, H = grid.shape
    Q = np.zeros((W, H, 4))  # 4 for ['up', 'down', 'left', 'right']
    actions = ['up', 'down', 'left', 'right']
    
    def get_action(x, y):
        if random.uniform(0, 1) < epsilon:
            return random.choice(actions)
        else:
            return actions[np.argmax(Q[x, y])]
    
    for _ in range(episodes):
        x, y = random.randint(0, W-1), random.randint(0, H-1)
        while grid[x, y] == 0:
            action = get_action(x, y)
            if action == 'up':
                new_x = max(x - 1, 0)
                new_y = y
            elif action == 'down':
                new_x = min(x + 1, W - 1)
                new_y = y
            elif action == 'left':
                new_x = x
                new_y = max(y - 1, 0)
            elif action == 'right':
                new_x = x
                new_y = min(y + 1, H - 1)
            
            reward = grid[new_x, new_y]
            best_next_action = np.argmax(Q[new_x, new_y])
            td_target = reward + gamma * Q[new_x, new_y, best_next_action] - r
            td_error = td_target - Q[x, y, actions.index(action)]
            Q[x, y, actions.index(action)] += alpha * td_error
            
            x, y = new_x, new_y
    
    policy = np.array([[actions[np.argmax(Q[x, y])] for y in range(H)] for x in range(W)])
    V = np.array([[np.max(Q[x, y]) for y in range(H)] for x in range(W)])
    
    return V, policy




# Run Bellman Iteration
V_bellman = bellman_equation(grid, p, r, gamma)
print("Bellman Value Function:")
print(V_bellman)

# Run Value Iteration (Model-Based RL)
V_value_iter, policy_value_iter = value_iteration(grid, p, r, gamma)
print("Value Iteration Value Function:")
print(V_value_iter)
print("Value Iteration Policy:")
print(policy_value_iter)

# Run Q-Learning (Model-Free RL)
V_q, policy_q = q_learning(grid, p, r, gamma)
print("Q-Learning Value Function:")
print(V_q)
print("Q-Learning Policy:")
print(policy_q)
