import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import curses
import numpy as np
from curses_stuff import draw_grid

map = [
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'X', 'X', 'X', 'X', 'O', 'O', 'O'],
    ['X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['X', 'O', 'X', 'X', 'X', 'X', 'X', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'X', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ]
VISUALIZATION = False
goal = (7, 7)
start = (0, 0)
min_epsilon = 0.1 # higher values set a higher floor for the probability of the agent choosing a random action
learning_rate = 0.1 # higher values make the agent more likely to update its Q-table with new information
decay_rate = .0001 # higher values make the epsilon curve sharper, decreasing faster, leading to less exploration quicker
gamma = 0.95 # higher values make the agent more likely to choose actions that lead to higher rewards
shaping_factor = .01 # higher values give the agent higher reward for choosing a spot closer to the goal
num_iterations = 3_000

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

def final_run(q_table):
    state = pos_to_state(start)
    step_count = 0
    while state != pos_to_state(goal):
        if step_count > 100:
            print("Agent couldn't reach goal")
            return
        step_count += 1
        action = np.argmax(q_table[state])
        _, state = update_state(state, action)
    print(f"Final run took {step_count} steps")

def main(stdscr):
    start_time = datetime.datetime.now()
    figure, axes = plt.subplots(2, 1)
    state = pos_to_state(start)
    map[goal[0]][goal[1]] = 'G'
    if VISUALIZATION:
        curses.curs_set(0)  # Hide the cursor
    steps_to_goal = 0
    for i in range(num_iterations):
        # print(i, end='\r')
        if VISUALIZATION:
            draw_grid(stdscr, map, state_to_pos(state))
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-decay_rate * i)
        action = epsilon_greedy(epsilon, q_table, state)
        reward, new_state = update_state(state, action)
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * q_table[new_state].max() - q_table[state, action])
        if new_state == pos_to_state(goal):
            axes[1].scatter(i, steps_to_goal)
            state = 0
            steps_to_goal = 0
        else:
            state = new_state
            steps_to_goal += 1
        if VISUALIZATION:
            time.sleep(.01)
        axes[0].scatter(i, reward)
    print(f"Training took {datetime.datetime.now()-start_time}")
    final_run(q_table)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Reward")
    axes[0].set_xlim(0, num_iterations)
    axes[0].set_ylim(-1, 1)
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Steps to goal")
    axes[1].set_xlim(0, num_iterations)
    plt.tight_layout()
    plt.show()

if VISUALIZATION:
    curses.wrapper(main)
else:
    main(None)