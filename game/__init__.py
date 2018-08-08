import numpy as np

def is_empty(states, x, y):
    for z in range(states.shape[2]):
        if states[x][y][z] != 0:
            return False
    return True

states = np.zeros([12,12,6])
states[2][2][1] = 1
print(is_empty(states, 1, 1))
print(is_empty(states, 2, 2))

