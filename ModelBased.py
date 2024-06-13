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

V, policy = value_iteration(grid, p, r, gamma)
print("Value Function:")
print(V)
print("Policy:")
print(policy)