import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import curses
import numpy as np
from curses_stuff import draw_grid
import itertools
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
import plotly.graph_objects as go

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
VISUALIZATION = True
goal = (7, 7)
start = (0, 0)

def epsilon_greedy(epsilon, q_table, state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state])
    
def update_state(state, action, hyperparameters, episode_history=None, steps_to_goal=0):
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
    if out_of_bounds(row, col) or map[row][col] == 'X' or (episode_history is not None and new_state in episode_history):
        return -1, state
    elif map[row][col] == 'G':
        return 1, new_state
    else:
        return shaping_reward(state, new_state, pos_to_state(goal), episode_history, steps_to_goal, hyperparameters), new_state
    
def agent_is_stuck(state, map, episode_history):
    position = state_to_pos(state)
    for i,j in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if not out_of_bounds(position[0] + i, position[1] + j) and pos_to_state((position[0] + i, position[1] + j)) not in episode_history and map[position[0] + i][position[1] + j] != 'X':
            return False
    return True
    
def shaping_reward(state, new_state, goal_state, episode_history, steps_to_goal, hyperparameters):
    old_distance = manhattan_distance(state, goal_state)
    new_distance = manhattan_distance(new_state, goal_state)
    retracing_penalty = 0
    return (old_distance - new_distance) * hyperparameters['distance_shaping_factor'] + (-retracing_penalty if episode_history is not None and new_state in episode_history else 0) * steps_to_goal

def state_to_pos(state):
    return (state // 8, state % 8)
def pos_to_state(pos):
    return pos[0] * 8 + pos[1]
def out_of_bounds(row, col):
    return row < 0 or col < 0 or row >= 8 or col >= 8
def manhattan_distance(state1: int, state2: int) -> int:
    # Convert state to grid positions (row, column)
    row1, col1 = state_to_pos(state1)
    row2, col2 = state_to_pos(state2)
    
    # Calculate and return the Manhattan distance
    return abs(row1 - row2) + abs(col1 - col2)

def final_run(q_table, hyperparameters):
    state = pos_to_state(start)
    step_count = 0
    while state != pos_to_state(goal):
        if step_count > 100:
            print("Agent couldn't reach goal")
            return
        step_count += 1
        action = np.argmax(q_table[state])
        _, state = update_state(state, action, hyperparameters)
    print(f"Final run took {step_count} steps")


def train(q_table, hyperparameters, plot=False, stdscr=None):
    state = pos_to_state(start)
    map[goal[0]][goal[1]] = 'G'
    if VISUALIZATION:
        curses.curs_set(0)  # Hide the cursor
    steps_to_goal = 0
    episode_history = set()
    steps_arr = []
    reward_arr = []
    for i in range(hyperparameters['num_iterations']):
        # print(i, end='\r')
        if VISUALIZATION:
            draw_grid(stdscr, map, state_to_pos(state))
        epsilon = hyperparameters['min_epsilon'] + (1 - hyperparameters['min_epsilon']) * np.exp(-hyperparameters['decay_rate'] * i)
        action = epsilon_greedy(epsilon, q_table, state)
        reward, new_state = update_state(state, action, hyperparameters, episode_history, steps_to_goal)
        q_table[state, action] = q_table[state, action] + hyperparameters['learning_rate'] * (reward + hyperparameters['gamma'] * q_table[new_state].max() - q_table[state, action])
        if new_state == pos_to_state(goal) or agent_is_stuck(state, map, episode_history):
            state = 0
            episode_history = set()
            if new_state == pos_to_state(goal):
                steps_arr.append([i, steps_to_goal])
            steps_to_goal = 0
        else:
            state = new_state
            steps_to_goal += 1
            episode_history.add(state)
        if VISUALIZATION:
            time.sleep(.01)
        if plot:
            reward_arr.append([i, reward])

    # calculate average steps to goal
    if len(steps_arr) == 0:
        steps_avg = np.inf
    else:
        _, steps = zip(*steps_arr)
        steps_avg = np.mean(steps)
    return steps_avg, reward_arr, steps_arr

def execute_grid_search():
    hyperparameters = {
        'learning_rate': [0.5, 0.55, 0.6, .65, .7], # bad: 0.01, 0.1
        'decay_rate': [0.01, 0.1, 0.5], # bad: 0.0001, 0.001
        'gamma': [0.9, 0.95, 0.99],
        'distance_shaping_factor': [0.01, 0.1, 1],
        'retracing_penalty': [0.1, 0.3, 0.5],
        'min_epsilon': [0.0001, 0.001, 0.01], # bad:
        'num_iterations': [3_000],
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_performance = float('inf')
    best_params = None

    start_time = datetime.datetime.now()
    # Execute grid search in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_train, combinations))

    for combination, avg_steps in results:
        if avg_steps < best_performance:
            best_performance = avg_steps
            best_params = combination

    print(f"Grid search took {datetime.datetime.now()-start_time}")
    print("Best Hyperparameters:", best_params)
    print("Best Average Steps to Goal:", best_performance)

    # 3d plot of decay_rate, learning_rate, and gamma with color representing average steps to goal
    create_3d_plot_of_hyperparameters(results, 'learning_rate', 'decay_rate', 'min_epsilon', True)

def parallel_train(combination):
    q_table = np.zeros((len(map)*len(map[0]), 4))  # Reset Q-table for each combination
    avg_steps, _, _ = train(q_table, combination)
    return combination, avg_steps

def create_3d_plot_of_hyperparameters(results, var1, var2, var3, scale_colors=False):
    # Extracting hyperparameters and performances
    v1 = [result[0][var1] for result in results]
    v2 = [result[0][var2] for result in results]
    v3 = [result[0][var3] for result in results]
    avg_steps_to_goal = [result[1] for result in results]

    if scale_colors:
        # Scale avg_steps_to_goal from 0 to 100
        min_steps = min(avg_steps_to_goal)
        avg_steps_to_goal = [step if step != np.inf else 100 for step in avg_steps_to_goal]
        max_steps = np.mean(avg_steps_to_goal)

        print(f"Min: {min_steps}, Max: {max_steps}, Avg: {np.mean(avg_steps_to_goal)}, Std: {np.std(avg_steps_to_goal)}")
        scaled_steps_to_goal = [(min(step, max_steps) - min_steps) * (100.0 / max_steps) for step in avg_steps_to_goal]
        print(f"Scaled Min: {min(scaled_steps_to_goal)}, Scaled Max: {max(scaled_steps_to_goal)}, Scaled Avg: {np.mean(scaled_steps_to_goal)}, Scaled Std: {np.std(scaled_steps_to_goal)}")     
    
    # Plotting
    fig = go.Figure(data=[go.Scatter3d(
        x=v1,
        y=v2,
        z=v3,
        mode='markers',
        marker=dict(
            size=8,
            color=scaled_steps_to_goal if scale_colors else avg_steps_to_goal,  # set color to average steps to goal
            colorscale='Viridis',     # choose a colorscale
            opacity=0.8,
            colorbar=dict(title=f"{'Scaled ' if scale_colors else ''}Avg Steps to Goal")
        )
    )])

    fig.update_layout(
        title='Hyperparameter Optimization Results',
        scene=dict(
            xaxis_title=var1,
            yaxis_title=var2,
            zaxis_title=var3
        ),
        margin=dict(r=0, l=0, b=0, t=0)
    )
    fig.show()


def main(stdscr):
    hyperparameters = {
        'min_epsilon': 0.001, # higher values set a higher floor for the probability of the agent choosing a random action
        'learning_rate': 0.6, # higher values make the agent more likely to update its Q-table with new information
        'decay_rate': 0.01, # higher values make the epsilon curve sharper, decreasing faster, leading to less exploration quicker
        'gamma': 0.99, # higher values make the agent more likely to choose actions that lead to higher rewards
        'distance_shaping_factor': 0.1, # higher values give the agent higher reward for choosing a spot closer to the goal
        'retracing_penalty': 0.1, # higher values penalize the agent more for retracing its steps
        'num_iterations': 3_000,
    }
    # rows = state for all possible squares in the grid
    # cols = the four actions (up, down, left, right)
    q_table = np.zeros((len(map)*len(map[0]), 4))
    print(q_table.shape)
    figure, axes = plt.subplots(2, 1)
    start_time = datetime.datetime.now()
    _, reward_arr, steps_arr = train(q_table, hyperparameters, True, stdscr)
    print(f"Training took {datetime.datetime.now()-start_time}")
    x, y = zip(*reward_arr)
    axes[0].scatter(x,y)
    x, y = zip(*steps_arr)
    axes[1].scatter(x,y)
    final_run(q_table, hyperparameters)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Reward")
    axes[0].set_xlim(0, hyperparameters['num_iterations'])
    axes[0].set_ylim(-1, 1)
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Steps to goal")
    axes[1].set_xlim(0, hyperparameters['num_iterations'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if VISUALIZATION:
        curses.wrapper(main)
    else:
        main(None)
    # execute_grid_search()