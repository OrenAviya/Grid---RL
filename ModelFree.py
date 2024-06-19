import random
import numpy as np

# נתונים לדוגמה
W, H = 5, 5  # מימדי הלוח
L = [(1, 2, 10), (3, 4, -5), (2, 2, 0)]  # פרסים, עונשים וקירות
p = 0.8  # סיכוי הצלחת הצעד
r = -1  # עלות הצעד
gamma = 0.5  # גורם הנחה

# יצירת לוח המשחק
grid = np.zeros((W, H))
for (x, y, reward) in L:
    grid[x, y] = reward


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

V_q, policy_q = q_learning(grid, p, r, gamma)
print("Q-Learning Value Function:")
print(V_q)
print("Q-Learning Policy:")
print(policy_q)



# def q_learning(w, h, L, p, r, gamma=0.5, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, threshold=0.01, max_episodes=1000):
#     Q = np.zeros((w, h, 4))
#     for x, y, v in L:
#         if v == 0:
#             Q[x, y, :] = -10  # ערך Q נמוך מאוד לקירות

#     actions = ['up', 'down', 'left', 'right']
#     action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    
#     def is_valid_move(x, y, L, w, h):
#         return 0 <= x < w and 0 <= y < h and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

#     def get_action(x, y, epsilon):
#         valid_actions = [a for a in actions if is_valid_move(x + (a == 'down') - (a == 'up'), y + (a == 'right') - (a == 'left'), L, w, h)]
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
#         while True:
#             action = get_action(x, y, epsilon)
#             if action == 'none':
#                 break
            
#             if action == 'up':
#                 new_x, new_y = max(x - 1, 0), y
#             elif action == 'down':
#                 new_x, new_y = min(x + 1, w - 1), y
#             elif action == 'left':
#                 new_x, new_y = x, max(y - 1, 0)
#             elif action == 'right':
#                 new_x, new_y = x, min(y + 1, h - 1)

#             if not is_valid_move(new_x, new_y, L, w, h):
#                 new_x, new_y = x, y  # הישאר במקום אם המהלך לא חוקי

#             if random.uniform(0, 1) < p:
#                 x, y = new_x, new_y

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


# def q_learning(grid, p, r, gamma, alpha=0.1, epsilon=0.1, episodes=1000):
#     W, H = grid.shape
#     Q = np.zeros((W, H, 4))  # 4 for ['up', 'down', 'left', 'right']
#     actions = ['up', 'down', 'left', 'right']
    
#     def get_action(x, y):
#         if random.uniform(0, 1) < epsilon:
#             return random.choice(actions)
#         else:
#             return actions[np.argmax(Q[x, y])]
    
#     for _ in range(episodes):
#         x, y = random.randint(0, W-1), random.randint(0, H-1)
#         while grid[x, y] == 0:
#             action = get_action(x, y)
#             if action == 'up':
#                 new_x = max(x - 1, 0)
#                 new_y = y
#             elif action == 'down':
#                 new_x = min(x + 1, W - 1)
#                 new_y = y
#             elif action == 'left':
#                 new_x = x
#                 new_y = max(y - 1, 0)
#             elif action == 'right':
#                 new_x = x
#                 new_y = min(y + 1, H - 1)
            
#             reward = grid[new_x, new_y]
#             best_next_action = np.argmax(Q[new_x, new_y])
#             td_target = reward + gamma * Q[new_x, new_y, best_next_action] - r
#             td_error = td_target - Q[x, y, actions.index(action)]
#             Q[x, y, actions.index(action)] += alpha * td_error
            
#             x, y = new_x, new_y
    
#     policy = np.array([[actions[np.argmax(Q[x, y])] for y in range(H)] for x in range(W)])
#     V = np.array([[np.max(Q[x, y]) for y in range(H)] for x in range(W)])
    
#     return V, policy



# def q_learning(grid, p, r, gamma, alpha=0.01, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995, threshold=0.01, max_episodes=5000, max_steps=100):
#     W, H = grid.shape
#     Q = np.zeros((W, H, 4))
#     for x, y, v in L:
#         if v == 0:
#             Q[x, y, :] = -np.inf  # Negative infinity for walls to ensure not chosen

#     actions = ['up', 'down', 'left', 'right']
#     action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
#     moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

#     def is_valid_move(x, y, L, w, h):
#         return 0 <= x < h and 0 <= y < w and (x, y) not in [(gx, gy) for gx, gy, v in L if v == 0]

#     def get_action(x, y, epsilon):
#         valid_actions = [a for a in actions if is_valid_move(x + moves[a][0], y + moves[a][1], L, w, h)]
#         if not valid_actions:
#             return None
#         if random.uniform(0, 1) < epsilon:
#             return random.choice(valid_actions)
#         else:
#             valid_indices = [action_indices[a] for a in valid_actions]
#             return valid_actions[np.argmax(Q[x, y, valid_indices])]

#     def get_best_next_action(x, y, current_action):
#         valid_actions = [a for a in actions if is_valid_move(x + moves[a][0], y + moves[a][1], L, w, h)]
#         if not valid_actions:
#             return None
#         # Exclude the current action from consideration
#         valid_actions = [a for a in valid_actions if a != current_action]
#         if not valid_actions:  # No valid actions other than the current one
#             return None
#         valid_indices = [action_indices[a] for a in valid_actions]
#         return valid_actions[np.argmax(Q[x, y, valid_indices])]

#     epsilon = epsilon_start

#     for episode in range(max_episodes):
#         x, y = random.randint(0, h - 1), random.randint(0, w - 1)
#         for step in range(max_steps):
#             action = get_action(x, y, epsilon)
#             if action is None:
#                 break

#             new_x, new_y = x + moves[action][0], y + moves[action][1]
#             if not is_valid_move(new_x, new_y, L, w, h):
#                 new_x, new_y = x, y

#             if random.uniform(0, 1) < p:
#                 x, y = new_x, new_y
#             else:
#                 alternative_moves = ['left', 'right'] if action in ['up', 'down'] else ['up', 'down']
#                 alt_move = random.choice(alternative_moves)
#                 x_alt, y_alt = x + moves[alt_move][0], y + moves[alt_move][1]
#                 if is_valid_move(x_alt, y_alt, L, w, h):
#                     x, y = x_alt, y_alt

#             reward = next((v for gx, gy, v in L if gx == x and gy == y), r)
#             if reward != r:  # Reached a terminal state (reward or penalty)
#                 td_target = reward  # No need to consider future states for terminal states
#             else:  # Not in a terminal state
#                 best_next_action = get_best_next_action(x, y, action)
#                 if best_next_action is None:
#                     break  # No valid actions in the next state, end episode
#                 td_target = reward + gamma * Q[x, y, action_indices[best_next_action]]

#             td_error = td_target - Q[x, y, action_indices[action]]
#             Q[x, y, action_indices[action]] += alpha * td_error

#             epsilon = max(epsilon_end, epsilon_decay * epsilon)

#     V = np.max(Q, axis=2)
#     policy = np.array([[get_best_next_action(x, y, None) for y in range(H)] for x in range(W)])

#     return V, policy