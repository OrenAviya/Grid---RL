import numpy as np 
import random



# נתונים לדוגמה
# w = 5
# h = 5
# L = [(4,0,-10),(0,4,-10),(1,1,1),(3,3,2)]
# p = 0.9
# r = -0.5
# w = 12
# h = 6
# L = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,0)]
# p = 0.9
# r = -1

# w = 12
# h = 4
# L = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,0)]
# p = 1
# r = -1

w = 4
h = 3
L = [(1,1,0),(3,2,1),(3,1,-1)]
p = 0.8
r = -1
gamma = 0.5  # גורם הנחה
# w, h = 4, 3
# L = [(0, 1, 0), (1, 1, -10), (3, 2, 2)]
# p = 0.8
# r = -1
# # יצירת לוח המשחק
# grid = np.zeros((W, H))
# for (x, y, reward) in L:
#     grid[x, y] = reward

# W, H = 6, 6
# L = [(1, 1, 10), (4, 4, -10), (2, 2, 5), (3, 3, -5)]
# p = 0.8
# r = -1

# W, H = 4, 4
# L = [(0, 3, 10), (3, 0, -10)]
# p = 0.9
# r = -1
def create_grid(w, h, L):
    grid = np.zeros((w, h))
    for (x, y, reward) in L:
        grid[w - x - 1, y] = reward
    return grid

grid =  create_grid(w, h, L)


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


# # כרגע הכי טוב: 
# def q_learning(w, h, L, p, r, gamma, alpha=0.01, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, threshold=0.01, max_episodes=5000, max_steps=100):
#     Q = np.zeros((w, h, 4))
#     for x, y, v in L:
#         if v == 0:
#             Q[x, y, :] = -10  # ערך Q נמוך מאוד לקירות

#     actions = ['up', 'down', 'left', 'right']
#     action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
#     moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

#     def is_valid_move(x, y, L, w, h):
#         return 0 <= x < w and 0 <= y < h and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

#     def get_action(x, y, epsilon, last_action):
#         valid_actions = [a for a in actions if is_valid_move(x + moves[a][0], y + moves[a][1], L, w, h)]
#         if not valid_actions:
#             return 'none'
#         if random.uniform(0, 1) < epsilon:
#             return random.choice(valid_actions)
#         else:
#             valid_indices = [action_indices[a] for a in valid_actions]
#             best_action = actions[valid_indices[np.argmax(Q[x, y, valid_indices])]]
#             # מניעת לולאות אינסופיות
#             if best_action == last_action:
#                 valid_actions.remove(best_action)
#                 if valid_actions:
#                     best_action = random.choice(valid_actions)
#             return best_action

#     epsilon = epsilon_start

#     for episode in range(max_episodes):
#         x, y = random.randint(0, w - 1), random.randint(0, h - 1)
#         last_action = None
#         for step in range(max_steps):
#             action = get_action(x, y, epsilon, last_action)
#             if action == 'none':
#                 break
            
#             new_x, new_y = x + moves[action][0], y + moves[action][1]

#             if not is_valid_move(new_x, new_y, L, w, h):
#                 new_x, new_y = x, y  # הישאר במקום אם המהלך לא חוקי

#             if random.uniform(0, 1) < p:
#                 x, y = new_x, new_y
#             else:
#                 alternative_moves = ['left', 'right'] if action in ['up', 'down'] else ['up', 'down']
#                 alt_move = random.choice(alternative_moves)
#                 x_alt, y_alt = x + moves[alt_move][0], y + moves[alt_move][1]
#                 if is_valid_move(x_alt, y_alt, L, w, h):
#                     x, y = x_alt, y_alt

#             reward = next((v for gx, gy, v in L if gx == x and gy == y), r)
#             best_next_action = np.argmax(Q[x, y])
#             td_target = reward + gamma * Q[x, y, best_next_action]
#             td_error = td_target - Q[x, y, action_indices[action]]
#             Q[x, y, action_indices[action]] += alpha * td_error

#             if reward != r:
#                 break

#             if np.max(np.abs(td_error)) < threshold:
#                 break

#             last_action = action

#         epsilon = max(epsilon_end, epsilon_decay * epsilon)
    
#     def q_values_to_policy(V, L, w, h):
#         """
#         Converts a Value table into a policy, choosing the action leading to the highest Q-value.

#         Returns:
#             np.ndarray: The policy matrix (w x h), where each element is the action ('up', 'down', 'left', 'right')
#                         leading to the highest V-value for that state.
#         """

#         actions = ['up', 'down', 'left', 'right']
#         moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

#         def is_valid_move(x, y):
#             return 0 <= x < w and 0 <= y < h and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

#         policy = np.empty((w, h), dtype=object)

#         for x in range(w):
#             for y in range(h):
#                 if not is_valid_move(x, y):
#                     policy[x, y] = 'Exit'  # Or any other indicator for invalid state
#                     continue

#                 max_value = -np.inf
#                 best_action = None
#                 for action in actions:
#                     next_x, next_y = x + moves[action][0], y + moves[action][1]
#                     if is_valid_move(next_x, next_y) and V[next_x, next_y] > max_value:
#                         max_value = V[next_x, next_y]
#                         best_action = action
#                 policy[x, y] = best_action

#         return policy

#     # policy = np.array([[get_action(x, y, 0, None) for y in range(h)] for x in range(w)])
#     V = np.array([[np.max(Q[x, y]) if is_valid_move(x, y, L, w, h) else -10 for y in range(h)] for x in range(w)])
#     policy = q_values_to_policy(V  ,L,w,h)

#     return V, policy

# def q_learning(w, h, L, p, r, gamma=0.9, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999, threshold=0.01, max_episodes=8000, max_steps=100):
#     Q = np.zeros((w, h, 4))
#     for x, y, v in L:
#         if v == 0:
#             Q[x, y, :] = -10  # ערך Q נמוך מאוד לקירות

#     actions = ['up', 'down', 'left', 'right']
#     action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
#     moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

#     def is_valid_move(x, y, L, w, h):
#         return 0 <= x < w and 0 <= y < h and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

#     def get_action(x, y, epsilon):
#         valid_actions = [a for a in actions if is_valid_move(x + moves[a][0], y + moves[a][1], L, w, h)]
#         if not valid_actions:
#             return 'none'
#         if random.uniform(0, 1) < epsilon:
#             return random.choice(valid_actions)
#         else:
#             valid_indices = [action_indices[a] for a in valid_actions]
#             return actions[valid_indices[np.argmax(Q[x, y, valid_indices])]]

#     epsilon = epsilon_start

#     for episode in range(max_episodes):
#         x, y = random.randint(0, w - 1), random.randint(0, h - 1)
#         for step in range(max_steps):
#             action = get_action(x, y, epsilon)
#             if action == 'none':
#                 break
            
#             new_x, new_y = x + moves[action][0], y + moves[action][1]

#             if not is_valid_move(new_x, new_y, L, w, h):
#                 new_x, new_y = x, y  # הישאר במקום אם המהלך לא חוקי

#             if random.uniform(0, 1) < p:
#                 x, y = new_x, new_y
#             else:
#                 alternative_moves = ['left', 'right'] if action in ['up', 'down'] else ['up', 'down']
#                 alt_move = random.choice(alternative_moves)
#                 x_alt, y_alt = x + moves[alt_move][0], y + moves[alt_move][1]
#                 if is_valid_move(x_alt, y_alt, L, w, h):
#                     x, y = x_alt, y_alt

#             reward = next((v for gx, gy, v in L if gx == x and gy == y), r)
#             best_next_action = np.argmax(Q[x, y])
#             td_target = reward + gamma * Q[x, y, best_next_action]
#             td_error = td_target - Q[x, y, action_indices[action]]
#             Q[x, y, action_indices[action]] += alpha * td_error

#             if reward != r:
#                 break

#             if np.max(np.abs(td_error)) < threshold:
#                 break

#         epsilon = max(epsilon_end, epsilon_decay * epsilon)

#     policy = np.array([[get_action(x, y, 0) for y in range(h)] for x in range(w)])
#     V = np.array([[np.max(Q[x, y]) if is_valid_move(x, y, L, w, h) else -10 for y in range(h)] for x in range(w)])

#     return V, policy

# def q_learning(w, h, L, p, r, gamma=0.5, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999, threshold=0.01, max_episodes=8000, max_steps=100):
#     Q = np.zeros((w, h, 4))
#     for x, y, v in L:
#         if v == 0:
#             Q[x, y, :] = -10  # ערך Q נמוך מאוד לקירות

#     actions = ['up', 'down', 'left', 'right']
#     action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
#     moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

#     def is_valid_move(x, y, L, w, h):
#         return 0 <= x < w and 0 <= y < h and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

#     def get_action(x, y, epsilon):
#         valid_actions = [a for a in actions if is_valid_move(x + moves[a][0], y + moves[a][1], L, w, h)]
#         if not valid_actions:
#             return 'none'
#         if random.uniform(0, 1) < epsilon:
#             return random.choice(valid_actions)
#         else:
#             valid_indices = [action_indices[a] for a in valid_actions]
#             return actions[valid_indices[np.argmax(Q[x, y, valid_indices])]]

#     epsilon = epsilon_start

#     for episode in range(max_episodes):
#         x, y = random.randint(0, w - 1), random.randint(0, h - 1)
#         for step in range(max_steps):
#             action = get_action(x, y, epsilon)
#             if action == 'none':
#                 break
            
#             new_x, new_y = x + moves[action][0], y + moves[action][1]

#             if not is_valid_move(new_x, new_y, L, w, h):
#                 new_x, new_y = x, y  # הישאר במקום אם המהלך לא חוקי

#             if random.uniform(0, 1) < p:
#                 x, y = new_x, new_y
#             else:
#                 alternative_moves = ['left', 'right'] if action in ['up', 'down'] else ['up', 'down']
#                 alt_move = random.choice(alternative_moves)
#                 x, y = x + moves[alt_move][0], y + moves[alt_move][1]
#                 if not is_valid_move(x, y, L, w, h):
#                     x, y = new_x, new_y

#             reward = next((v for gx, gy, v in L if gx == x and gy == y), r)
#             best_next_action = np.argmax(Q[x, y])
#             td_target = reward + gamma * Q[x, y, best_next_action]
#             td_error = td_target - Q[x, y, action_indices[action]]
#             Q[x, y, action_indices[action]] += alpha * td_error

#             if reward != r:
#                 break

#             if np.max(np.abs(td_error)) < threshold:
#                 break

#         epsilon = max(epsilon_end, epsilon_decay * epsilon)

#     policy = np.array([[get_action(x, y, 0) for y in range(h)] for x in range(w)])
#     V = np.array([[np.max(Q[x, y]) if is_valid_move(x, y, L, w, h) else -10 for y in range(h)] for x in range(w)])

#     return V, policy

def q_learning(grid, p, r, gamma, alpha=0.01, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995, threshold=0.01, max_episodes=5500, max_steps=100):
    W, H = grid.shape
    Q = np.zeros((W, H, 4))
    for x, y, v in L:
        if v == 0:
            Q[x, y, :] = -np.inf  # Negative infinity for walls to ensure not chosen

    actions = ['up', 'down', 'left', 'right']
    action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def is_valid_move(x, y, L, w, h):
        return 0 <= x < w and 0 <= y < h and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

    def get_action(x, y, epsilon):
        valid_actions = [a for a in actions if is_valid_move(x + moves[a][0], y + moves[a][1], L, w, h)]
        if not valid_actions:
            return None  
        if random.uniform(0, 1) < epsilon:
            return random.choice(valid_actions)
        else:
            valid_indices = [action_indices[a] for a in valid_actions]
            return valid_actions[np.argmax(Q[x, y, valid_indices])]  

    def get_best_next_action(x, y, current_action):
        valid_actions = [a for a in actions if is_valid_move(x + moves[a][0], y + moves[a][1], L, w, h)]
        if not valid_actions:
            return None
        # Exclude the current action from consideration
        valid_actions = [a for a in valid_actions if a != current_action] 
        if not valid_actions:  # No valid actions other than the current one
            return None
        valid_indices = [action_indices[a] for a in valid_actions]
        return valid_actions[np.argmax(Q[x, y, valid_indices])]

    epsilon = epsilon_start

    for episode in range(max_episodes):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        for step in range(max_steps):
            action = get_action(x, y, epsilon)
            if action is None:
                break

            new_x, new_y = x + moves[action][0], y + moves[action][1]
            if not is_valid_move(new_x, new_y, L, w, h):
                new_x, new_y = x, y  

            if random.uniform(0, 1) < p:
                x, y = new_x, new_y
            else:
                alternative_moves = ['left', 'right'] if action in ['up', 'down'] else ['up', 'down']
                alt_move = random.choice(alternative_moves)
                x_alt, y_alt = x + moves[alt_move][0], y + moves[alt_move][1]
                if is_valid_move(x_alt, y_alt, L, w, h):
                    x, y = x_alt, y_alt

            reward = next((v for gx, gy, v in L if gx == x and gy == y), r)
            if reward != r:  # Reached a terminal state (reward or penalty)
                td_target = reward  # No need to consider future states for terminal states
            else:  # Not in a terminal state
                best_next_action = get_best_next_action(x, y, action)
                if best_next_action is None:
                    break  # No valid actions in the next state, end episode
                td_target = reward + gamma * Q[x, y, action_indices[best_next_action]]
            
            td_error = td_target - Q[x, y, action_indices[action]]
            Q[x, y, action_indices[action]] += alpha * td_error

            epsilon = max(epsilon_end, epsilon_decay * epsilon) 

    policy = np.array([[get_best_next_action(x, y, None) for y in range(h)] for x in range(w)])
    V = np.array([[np.max(Q[x, y]) if is_valid_move(x, y, L, w, h) else -np.inf for y in range(h)] for x in range(w)])

    return V, policy

def q_values_to_policy(V, L, w, h):
    """
    Converts a Value table into a policy, choosing the action leading to the highest Q-value.

    Args:
        V (np.ndarray): The Value table (w x h).
        L (list): List of tuples representing (x, y, reward) for rewards, penalties, and walls.
        w (int): Width of the grid.
        h (int): Height of the grid.

    Returns:
        np.ndarray: The policy matrix (w x h), where each element is the action ('up', 'down', 'left', 'right')
                    leading to the highest V-value for that state.
    """

    actions = ['up', 'down', 'left', 'right']
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def is_valid_move(x, y):
        return 0 <= x < w and 0 <= y < h and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

    policy = np.empty((w, h), dtype=object)

    for x in range(w):
        for y in range(h):
            if not is_valid_move(x, y):
                policy[x, y] = 'Exit'  # Or any other indicator for invalid state
                continue

            max_value = -np.inf
            best_action = None
            for action in actions:
                next_x, next_y = x + moves[action][0], y + moves[action][1]
                if is_valid_move(next_x, next_y) and V[next_x, next_y] > max_value:
                    max_value = V[next_x, next_y]
                    best_action = action
            policy[x, y] = best_action

    return policy 


def print_reversed_values(V):
    """Prints the rows of a NumPy array in reversed order."""
    for row in V[::-1]:  # Iterate over rows in reverse order
        print(row)
 
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
V, policy = q_learning(grid, p, r, gamma)
# V, policy = q_learning(w, h, L , p, r,gamma)
print("Q-Learning Value Function:")
print_reversed_values(V)
Q_values = V[::-1] 
policy = q_values_to_policy(Q_values, L, w, h)
print("Q-Learning Policy:")
print(policy)


