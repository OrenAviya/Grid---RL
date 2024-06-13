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