import numpy as np

def create_grid(h, w, L):
    t=h
    h=w
    w=t
    grid = np.zeros((h, w))
    for (x, y, reward) in L:
        grid[h-x-1, y] = reward

    # # נבצע טרנספוזיציה
    transposed_grid = np.transpose(grid)
    transposed_grid = transposed_grid[::-1]
    transposed_grid = transposed_grid[:, ::-1]
    return transposed_grid

# נבנה מטריצה לדוגמה
h = 3
w = 4
L = [(1, 1, -100), (3, 2, 1), (3, 1, -1)]

# ניצור את המטריצה
grid = create_grid(h, w, L)

# הצגת התוצאות
print(grid)
