import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import curses
from tqdm.notebook import trange
from scipy.spatial.distance import cdist
import numpy as np
from numpy.typing import NDArray

from curses_stuff import draw_grid

map = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ]

goal = (7, 7)
start = (0, 0)
min_epsilon = 0.1 #probability of exploration action
learning_rate = 0.1
decay_rate = 0.0005
gamma = 0.95
shaping_factor = .1

# rows = state for all possible squares in the grid
# cols = the four actions (up, down, left, right)
q_table = np.zeros((64, 4))

def epsilon_greedy(epsilon, q_table, state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state])
    
def update_state(state, action):
    row, col = state_to_pos(state)
    if action == 0: #up
        row -= 1
    elif action == 1: #down
        row += 1
    elif action == 2: #left
        col -= 1
    elif action == 3: #right
        col += 1
    new_state = pos_to_state((row, col))
    if row < 0 or row > 7 or col < 0 or col > 7 or map[row][col] == 'X':
        return -1, state
    elif map[row][col] == 'G':
        return 1, new_state
    else:
        return shaping_reward(state, new_state, pos_to_state(goal)), new_state
    
def manhattan_distance(state1: int, state2: int) -> int:
    # Convert state to grid positions (row, column)
    row1, col1 = state_to_pos(state1)
    row2, col2 = state_to_pos(state2)
    
    # Calculate and return the Manhattan distance
    return abs(row1 - row2) + abs(col1 - col2)

def shaping_reward(state, new_state, goal_state):
    old_distance = manhattan_distance(state, goal_state)
    new_distance = manhattan_distance(new_state, goal_state)
    return (old_distance - new_distance) * shaping_factor

def state_to_pos(state):
    return (state // 8, state % 8)
def pos_to_state(pos):
    return pos[0] * 8 + pos[1]

def main(stdscr):
    state = pos_to_state(start)
    map[goal[0]][goal[1]] = 'G'
    num_iterations = 3_000
    figure, axes = plt.subplots(2, 1)
    steps_to_goal = 0
    curses.curs_set(0)  # Hide the cursor
    for i in range(num_iterations):
        # draw_grid(stdscr, map, state_to_pos(state))
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-decay_rate * i)
        action = epsilon_greedy(epsilon, q_table, state)
        reward, new_state = update_state(state, action)
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * q_table[new_state].max() - q_table[state, action])
        if new_state == 63:
            state = 0
            axes[1].scatter(i, steps_to_goal)
            steps_to_goal = 0
        else:
            state = new_state
        axes[0].scatter(i, reward)
        steps_to_goal += 1
        # time.sleep(.01)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Reward")
    axes[0].set_ylim(-1, 1)
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Steps to goal")
    axes[1].set_xlim(0, 3000)
    plt.tight_layout()
    plt.show()
curses.wrapper(main)