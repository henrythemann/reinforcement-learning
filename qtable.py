import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import curses
from tqdm.notebook import trange

map = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'G'],
    ]
min_epsilon = 0.1 #probability of exploration action
learning_rate = 0.1
decay_rate = 0.0005
gamma = 0.95

# rows = state for all possible squares in the grid
# cols = the four actions (up, down, left, right)
q_table = np.zeros((64, 4))

def draw_grid(stdscr, current_pos):
    # Clear screen
    stdscr.clear()
    
    height, width = 8, 8
    for i in range(height):
        for j in range(width):
            if (i, j) == current_pos:
                stdscr.addstr(i, j*2, "*")  # Mark current position
            elif map[i][j] == 'G': # goal
                stdscr.addstr(i, j*2, "G")
            elif map[i][j] == 'X': # obstacle
                stdscr.addstr(i, j*2, "X")
            else:
                stdscr.addstr(i, j*2, ".")  # Empty cell

    stdscr.refresh()

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
    if row < 0:
        return False
    if row > 7:
        return False
    if col < 0:
        return False
    if col > 7:
        return False
    return pos_to_state((row, col))

def state_to_pos(state):
    return (state // 8, state % 8)
def pos_to_state(pos):
    return pos[0] * 8 + pos[1]

def main(stdscr):
    state = 0
    num_iterations = 3000
    curses.curs_set(0)  # Hide the cursor
    figure, axes = plt.subplots(2, 1)
    steps_to_goal = 0
    for i in range(num_iterations):
        # draw_grid(stdscr, state_to_pos(state))
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-decay_rate * i)
        new_state = False
        while isinstance(new_state, bool) and new_state == False:
            action = epsilon_greedy(epsilon, q_table, state)
            new_state = update_state(state, action)
        reward = q_table[new_state].max()
        row, col = state_to_pos(new_state)
        if map[row][col] == 'X':
            reward = -1
        elif map[row][col] == 'G':
            reward = 1
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * q_table[new_state].max() - q_table[state, action])
        if new_state == 63:
            state = 0
            axes[1].scatter(i, steps_to_goal)
            steps_to_goal = 0
        else:
            state = new_state
        # time.sleep(.01)
        axes[0].scatter(i, reward)
        steps_to_goal += 1
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Reward")
    axes[0].set_ylim(-1, 1)
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Steps to goal")
    axes[1].set_xlim(0, 3000)
    plt.tight_layout()
    plt.show()
curses.wrapper(main)