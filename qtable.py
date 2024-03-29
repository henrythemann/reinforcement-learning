import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import curses
import numpy as np
import itertools
import argparse

def draw_grid(stdscr, map, current_pos):
    # Clear screen
    stdscr.clear()
    
    height, width = len(map), len(map[0])
    screen_height, screen_width = stdscr.getmaxyx()
    if not (height < screen_height and width*2 < screen_width):
        stdscr.addstr("Terminal window too small")
    else:
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
    
def update_state(state, action, maze, hyperparameters, episode_history=None, steps_to_goal=0):
    row, col = state_to_pos(state, maze.map())
    if action == 0: #up
        row -= 1
    elif action == 1: #down
        row += 1
    elif action == 2: #left
        col -= 1
    elif action == 3: #right
        col += 1
    new_state = pos_to_state((row, col), maze.map())
    if out_of_bounds(row, col, maze.map()) or maze.map()[row][col] == 'X' or (episode_history is not None and new_state in episode_history):
        return -1, state
    elif maze.map()[row][col] == 'G':
        return 1, new_state
    else:
        return shaping_reward(state, new_state, pos_to_state(maze.goal(), maze.map()), episode_history, steps_to_goal, maze.map(), hyperparameters), new_state
    
def agent_is_stuck(state, map, episode_history):
    position = state_to_pos(state, map)
    for i,j in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if not out_of_bounds(position[0] + i, position[1] + j, map) and pos_to_state((position[0] + i, position[1] + j), map) not in episode_history and map[position[0] + i][position[1] + j] != 'X':
            return False
    return True
    
def shaping_reward(state, new_state, goal_state, episode_history, steps_to_goal, map, hyperparameters):
    old_distance = manhattan_distance(state, goal_state, map)
    new_distance = manhattan_distance(new_state, goal_state, map)
    retracing_penalty = 0
    return (old_distance - new_distance) * hyperparameters['distance_shaping_factor'] + (-retracing_penalty if episode_history is not None and new_state in episode_history else 0) * steps_to_goal + steps_to_goal * -hyperparameters['step_penalty']

def state_to_pos(state, map):
    cols = len(map[0])
    return (state // cols, state % cols)

def pos_to_state(pos, map):
    cols = len(map[0])
    return pos[0] * cols + pos[1]

def out_of_bounds(row, col, map):
    rows = len(map)
    cols = len(map[0])
    return row < 0 or col < 0 or row >= rows or col >= cols

def manhattan_distance(state1: int, state2: int, map) -> int:
    # Convert state to grid positions (row, column)
    row1, col1 = state_to_pos(state1, map)
    row2, col2 = state_to_pos(state2, map)
    
    # Calculate and return the Manhattan distance
    return abs(row1 - row2) + abs(col1 - col2)

def final_run(q_table, maze, hyperparameters):
    state = pos_to_state(maze.start(), maze.map())
    step_count = 0
    while state != pos_to_state(maze.goal(), maze.map()):
        if step_count > 100:
            print("Agent couldn't reach goal")
            return
        step_count += 1
        action = np.argmax(q_table[state])
        _, state = update_state(state, action, maze, hyperparameters)
    print(f"Final run took {step_count} steps")


def train(maze, hyperparameters, num_iterations=3000, plot=False, stdscr=None):
    # rows = state for all possible squares in the grid
    # cols = the four actions (up, down, left, right)
    q_table = np.zeros((len(maze.map())*len(maze.map()[0]), 4))

    state = pos_to_state(maze.start(), maze.map())
    if stdscr is not None:
        curses.curs_set(0)  # Hide the cursor
    steps_to_goal = 0
    min_steps_to_goal = np.inf
    episode_history = set()
    steps_arr = []
    reward_arr = []
    for i in range(num_iterations):
        # print(i, end='\r')
        if stdscr is not None:
            draw_grid(stdscr, maze.map(), state_to_pos(state, maze.map()))
        epsilon = hyperparameters['min_epsilon'] + (1 - hyperparameters['min_epsilon']) * np.exp(-hyperparameters['decay_rate'] * i)
        action = epsilon_greedy(epsilon, q_table, state)
        reward, new_state = update_state(state, action, maze, hyperparameters, episode_history, steps_to_goal)
        q_table[state, action] = q_table[state, action] + hyperparameters['learning_rate'] * (reward + hyperparameters['gamma'] * q_table[new_state].max() - q_table[state, action])
        if new_state == pos_to_state(maze.goal(), maze.map()) or agent_is_stuck(state, maze.map(), episode_history):
            state = 0
            episode_history = set()
            if new_state == pos_to_state(maze.goal(), maze.map()):
                steps_arr.append([i, steps_to_goal])
                if steps_to_goal < min_steps_to_goal:
                    min_steps_to_goal = steps_to_goal
            steps_to_goal = 0
        else:
            state = new_state
            steps_to_goal += 1
            episode_history.add(state)
        if stdscr is not None:
            time.sleep(.01)
        if plot:
            reward_arr.append([i, reward])

    # calculate average steps to goal
    if len(steps_arr) == 0:
        steps_avg = np.inf
    else:
        _, steps = zip(*steps_arr)
        steps_avg = np.mean(steps)
    return steps_avg, min_steps_to_goal, reward_arr, steps_arr, q_table

def execute_grid_search(maze, num_iterations):
    hyperparameters = {
        'min_epsilon': [0.001, 0.01], # higher values set a higher floor for the probability of the agent choosing a random action
        'learning_rate': [0.5, 0.55, 0.6, .65, .7], # bad: 0.01, 0.1
        'decay_rate': [0.01], # higher values make the epsilon curve sharper, decreasing faster, leading to less exploration quicker
        'gamma': [0.9, 0.95, 0.99],
        'distance_shaping_factor': np.linspace(0.001, 0.1, 10),
        'retracing_penalty': np.linspace(0.01, 0.1, 5),
        'min_epsilon': [0.0001, 0.001, 0.01], # bad:
        'step_penalty': np.linspace(0.0001, 0.001, 5),
    }
    # add parallel_train function parameters to hyperparameters dict so we can pass them to the function
    hyperparameters['num_iterations'] = [num_iterations]
    hyperparameters['maze'] = [maze]
    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_performance = float('inf')
    best_params = None

    start_time = datetime.datetime.now()
    # Execute grid search in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_train_wrapper, combinations))

    for combination, var_to_minimize in results:
        if var_to_minimize < best_performance:
            best_performance = var_to_minimize
            best_params = combination

    print(f"Grid search took {datetime.datetime.now()-start_time}")
    print("Best Hyperparameters:", best_params)
    print("Best Steps Parameter:", best_performance)

    # 3d plot of decay_rate, learning_rate, and gamma with color representing average steps to goal
    create_3d_plot_of_hyperparameters(results, 'learning_rate', 'decay_rate', 'min_epsilon', True)

def parallel_train_wrapper(args):
    maze = args.pop('maze')
    num_iterations = args.pop('num_iterations')
    return parallel_train(maze, num_iterations, args)


def parallel_train(maze, num_iterations, combination):
    avg_steps, min_steps_to_goal, _, _, _ = train(maze, combination, num_iterations)
    var_to_minimize = (min_steps_to_goal + avg_steps) / 2
    return combination, var_to_minimize

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


class Maze:
    def __init__(self, map_file):
        self.__map = self.__load_map(map_file)
        self.__start = None
        self.__goal = None
        for i in range(len(self.__map)):
            for j in range(len(self.__map[i])):
                if self.__map[i][j] == 'S':
                    self.__start = (i, j)
                if self.__map[i][j] == 'G':
                    self.__goal = (i, j)
        if self.__start is None or self.__goal is None:
            raise ValueError("Map must contain a start and goal")
        
    def __load_map(self, file):
        with open(file, 'r') as f:
            return [list(line.strip().replace(' ','')) for line in f]
        
    def start(self):
        return self.__start
    
    def goal(self):
        return self.__goal
    
    def map(self):
        return self.__map

def setup_and_train(maze, num_iterations, stdscr=None):
    hyperparameters = {
        'min_epsilon': 0.001, # higher values set a higher floor for the probability of the agent choosing a random action
        'learning_rate': 0.6, # higher values make the agent more likely to update its Q-table with new information
        'decay_rate': 0.01, # higher values make the epsilon curve sharper, decreasing faster, leading to less exploration quicker
        'gamma': 0.99, # higher values make the agent more likely to choose actions that lead to higher rewards
        'distance_shaping_factor': 0.1, # higher values give the agent higher reward for choosing a spot closer to the goal
        'retracing_penalty': 0.1, # higher values penalize the agent more for retracing its steps
        'step_penalty': 0.0005, # higher values penalize the agent more for taking more steps
    }
    hyperparameters = {'min_epsilon': 0.001, 'learning_rate': 0.65, 'decay_rate': 0.01, 'gamma': 0.9, 'distance_shaping_factor': 0.045, 'retracing_penalty': 0.01, 'step_penalty': 0.0001}
    
    figure, axes = plt.subplots(2, 1)
    start_time = datetime.datetime.now()
    _, min_steps_to_goal, reward_arr, steps_arr, q_table = train(maze, hyperparameters, num_iterations, True, stdscr)
    print(f"Training took {datetime.datetime.now()-start_time}")
    print(f"Min steps to goal: {min_steps_to_goal}")
    if len(reward_arr) > 0:
        x, y = zip(*reward_arr)
        axes[0].scatter(x,y)
    if len(steps_arr) > 0:
        x, y = zip(*steps_arr)
        axes[1].scatter(x,y)
    final_run(q_table, maze, hyperparameters)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Reward")
    axes[0].set_xlim(0, num_iterations)
    axes[0].set_ylim(-1, 1)
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Steps to goal")
    axes[1].set_xlim(0, num_iterations)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='MAZE NAVIGATION')
    
    parser.add_argument('-g', '--grid_search', action='store_true', help='Execute grid search to find optimal hyperparameters')
    parser.add_argument('-v', '--visualize', action='store_true', help='Enable visualization of the agent training & navigating the maze')
    parser.add_argument('-i', '--num_iterations', type=int, default=3_000, help='Number of iterations to train the agent')
    parser.add_argument('-m', '--map', type=str, help='File containing the map')
    
    args = parser.parse_args()
    
    # error checking
    if args.grid_search and args.visualize:
        print("Cannot execute grid search and visualize at the same time")
        return
    if args.map is None:
        print("Please provide a map file")
        return

    num_iterations = args.num_iterations
    maze = Maze(args.map)

    global VISUALIZATION
    VISUALIZATION = False
    if args.grid_search:
        global ProcessPoolExecutor
        global go
        from concurrent.futures import ProcessPoolExecutor
        import plotly.graph_objects as go
        execute_grid_search(maze, num_iterations)
        return
    if args.visualize:
        VISUALIZATION = True
    if VISUALIZATION:
        curses.wrapper(lambda stdcr: setup_and_train(maze, num_iterations, stdcr))
    else:
        setup_and_train(maze, num_iterations)

if __name__ == "__main__":
    main()