import numpy as np
import pygame
import copy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from environment import WallE, PygameInit

GOOD_TRASH_REWARD = 250
GOAL_REWARD = 400
ENEMY_REWARD = -400
DEFAULT_REWARD = -1


# ---------------------------
# MySolver: value-iteration + policy extraction
# ---------------------------
class MySolver(WallE):
    DISCOUNT_FACTOR = 0.95

    def actions_model(self, reward_map):
        """
        Run Value Iteration on reward_map and produce:
        - policy_table
        - V_table
        """

        grid_size = self._WallE__grid_size
        gamma = self.DISCOUNT_FACTOR
        epsilon = 1e-4
        max_iterations = 50000
        convergence_history = []
        transition_table = self._WallE__calculate_transition_model(
            grid_size,
            self._WallE__probability_dict,
            reward_map
        )

        # initialize V_table
        V_table_map = np.zeros((grid_size, grid_size), dtype=float)
        #V_table_map = np.array(reward_map, dtype=float)
    
        # 1. value iteration
        for iteration in range(max_iterations):
            V_old = V_table_map.copy()
            delta = 0.0
            for r in range(grid_size):
                for c in range(grid_size):

                    state = (r, c)
                    tile, _ = self.grid[r][c]

                    if tile == 'B':
                        V_table_map[r, c] = -50
                        continue
                    if tile == 'F':
                        V_table_map[r, c] = reward_map[r][c]
                        continue

                        """ if tile == 'E':
                        V_table_map[r, c] = reward_map[r][c]
                        continue """

                    best_Q = -1e15
                    for action in range(4):
                        if action not in transition_table[state]:
                            continue

                        Q = 0.0
                        for prob, next_state, reward_next in transition_table[state][action]:
                            nr, nc = next_state
                            Q += prob * (reward_next + gamma * V_old[nr, nc])

                        if Q > best_Q:
                            best_Q = Q

                    V_table_map[r, c] = best_Q
                    delta = max(delta, abs(best_Q - V_old[r, c]))

            #delta = np.sum(np.abs(V_table_map - V_old))
            convergence_history.append(delta)
            if delta < epsilon:
                break

       #2. policy extraction
        policy = np.full((grid_size, grid_size), -1, dtype=np.int8)

        for r in range(grid_size):
            for c in range(grid_size):

                state = (r, c)
                tile, _ = self.grid[r][c]

                if tile in ('B', 'F'):
                    policy[r, c] = -1
                    continue

                best_Q = -1e15
                best_action = 0

                for action in range(4):

                    if action not in transition_table[state]:
                        continue

                    Q = 0.0
                    for prob, next_state, reward_next in transition_table[state][action]:
                        nr, nc = next_state
                        Q += prob * (reward_next + gamma * V_table_map[nr, nc])

                    if Q > best_Q:
                        best_Q = Q
                        best_action = action

                policy[r, c] = best_action

        return policy, V_table_map, convergence_history



def build_reward_map(env, current_threshold=5, is_goal_only=False):
    """
    Build Reward Map
    """
    grid_size = env._WallE__grid_size
    reward_map = [[DEFAULT_REWARD for _ in range(grid_size)] for _ in range(grid_size)]

    for r in range(grid_size):
        for c in range(grid_size):
            tile, val = env.grid[r][c]
            if tile == 'F':
                reward_map[r][c] = GOAL_REWARD
            if is_goal_only:
                continue
            elif tile == 'E':
                reward_map[r][c] = ENEMY_REWARD
            elif tile == 'B':
                reward_map[r][c] = -50
            elif tile == 'S':
                trash_value = val
                if trash_value <= current_threshold:
                    reward_map[r][c] = GOOD_TRASH_REWARD
                else:
                    relative_reward = (current_threshold / trash_value) * GOOD_TRASH_REWARD
                    reward_map[r][c] = relative_reward
            else:
                reward_map[r][c] = DEFAULT_REWARD
    return reward_map


def build_reward_map_for_mdp_key(env, initial_trash_points, threshold, trash_state_tuple):
    """
    Building the reward map based on Trash_Points
    """
    base_reward_map = build_reward_map(env, threshold, is_goal_only=False)
    mdp_reward_map = copy.deepcopy(base_reward_map)

    # For each trash that is removed, we set its Reward to DEFAULT_REWARD
    for i in range(len(initial_trash_points)):
        is_present = trash_state_tuple[i]
        tr, tc = initial_trash_points[i]
        if is_present == 0:
            mdp_reward_map[tr][tc] = DEFAULT_REWARD
    return mdp_reward_map


def precompute_policies(solver, initial_trash_points):
    """
    Precompute policies based on thresholds and Trash Points
    """
    THRESHOLDS = [5, 10, 20, 30, 40]
    policies_map = {}
    v_tables_map = {}
    print("--- Starting Precomputation of Policies ---")

    for threshold in THRESHOLDS:
        print(f"    -> Computing policy for Threshold: {threshold}")
        # All trash cases (8)
        for trash_state_tuple in itertools.product([0, 1], repeat=len(initial_trash_points)):
            reward_map = build_reward_map_for_mdp_key(solver, initial_trash_points, threshold, trash_state_tuple)
            policy_table, V_table,_ = solver.actions_model(reward_map)

            key = (int(threshold), tuple(trash_state_tuple))
            policies_map[key] = policy_table
            v_tables_map[key] = V_table
    return policies_map, v_tables_map



def get_current_trash_state_tuple(env, initial_trash_points):
    """
     Mapping the current state of the trash to an 8-tuple
    """
    current_trash_points = [tuple(p) for p in env.get_trash_points()]
    current_trash_set = set(current_trash_points)

    trash_state_tuple = []
    for point in initial_trash_points:
        if tuple(point) in current_trash_set:
            trash_state_tuple.append(1)
        else:
            trash_state_tuple.append(0)
    return tuple(trash_state_tuple)



def select_policy(env, policies_map,V_table_map, initial_trash_points, threshold):
    """
    Select an appropriate policy for the current environment state.
    """
    """if(moves>=90):
        return goal_table"""
    current_trash_tuple = get_current_trash_state_tuple(env, initial_trash_points)
    key = (threshold, current_trash_tuple)
    v_table = V_table_map.get(key)
    policy = policies_map.get(key)
    if current_trash_tuple == (0, 0, 0, 0, 0, 0, 0, 0):
        return goal_table, v_table
    """available_thresholds = sorted({k[0] for k in policies_map.keys()})
    zero_tuple = tuple(0 for _ in initial_trash_points)
    if current_trash_tuple == zero_tuple:
        for thr in available_thresholds:
            k = (thr, zero_tuple)
            if k in policies_map:
                return policies_map[k], v_table"""

    if policy is None:
        available_thresholds = sorted({k[0] for k in policies_map.keys()})
        if len(available_thresholds) == 0:
            return np.full((env._WallE__grid_size, env._WallE__grid_size), 0, dtype=np.int8), v_table

        closest_threshold = min(available_thresholds, key=lambda x: abs(x - threshold))
        fallback_key = (closest_threshold, current_trash_tuple)
        policy = policies_map.get(fallback_key)

        if policy is None:
            return np.full((env._WallE__grid_size, env._WallE__grid_size), 0, dtype=np.int8), v_table
    return policy , v_table
    """available_thresholds = sorted({k[0] for k in policies_map.keys()})
    if available_thresholds:
        closest = min(available_thresholds, key=lambda x: abs(x - threshold))
        fallback_key = (closest, tuple(current_trash_tuple))
        policy = policies_map.get(fallback_key)
        if policy is not None:
            return policy, v_table

    
    zero_tuple = tuple(0 for _ in initial_trash_points)
    if current_trash_tuple == zero_tuple:
        for thr in available_thresholds:
            k = (thr, zero_tuple)
            if k in policies_map:
                return policies_map[k], v_table"""
def visualize_convergence(history, title="Value Iteration Convergence"):
    """
    Plots the convergence history (max change in V per iteration).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linestyle='-', color='b', markersize=3)
    plt.title(title, fontsize=16)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Max Value Difference (epsilon)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.show()
def visualize_value_heatmap(V_table, env, title="Value Function Heatmap (Goal-Only Policy)"):
    """
    Heat map visualization for Goal only policies.
    """
    grid_size = V_table.shape[0]
    V_reversed = V_table[::-1]
    plt.figure(figsize=(8, 8))

    ax = sns.heatmap(
        V_reversed,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=.5,
        linecolor='black',
        cbar=True,
        yticklabels=True,
        xticklabels=True,
        annot_kws={"size": 10, "weight": "bold"}
    )
    for r in range(grid_size):
        for c in range(grid_size):
            tile, _ = env.grid[r][c]
           
            plot_r = grid_size - 1 - r
            
            if tile == 'F':
                ax.text(c + 0.5, plot_r + 0.5, 'GOAL',
                        ha='center', va='center', color='black',
                        fontsize=12, fontweight='heavy',
                        bbox={'facecolor': 'yellow', 'alpha': 0.6, 'pad': 2})

            elif tile == 'B':
                ax.text(c + 0.5, plot_r + 0.5, 'B',
                        ha='center', va='center', color='white',
                        fontsize=14, fontweight='heavy',
                        bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 2})
            elif tile == 'E':
                ax.text(c + 0.5, plot_r + 0.5, 'E',
                        ha='center', va='center', color='white',
                        fontsize=14, fontweight='heavy',
                        bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 2})

    plt.title(title, fontsize=16)
    plt.ylabel("Row (r)")
    plt.xlabel("Column (c)")
    plt.show()
def visualize_policy_arrows(policy_table, V_table, env, title="Policy and Value Heatmap"):
    """
    Combines Value Function Heatmap with Policy Arrows (Optimal Actions).
    """
    grid_size = policy_table.shape[0]
    V_reversed = V_table[::-1]

    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        V_reversed,
        cmap="YlGnBu",
        linewidths=.5,
        linecolor='black',
        cbar=True,
        annot=False, # Disable V-value annotations for cleaner arrows
        yticklabels=True,
        xticklabels=True
    )

    # Define arrow symbols for each action: 0:Up, 1:Down, 2:Left, 3:Right
    action_to_arrow = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    # Iterate through the grid to plot arrows and markers
    for r in range(grid_size):
        for c in range(grid_size):
            action = policy_table[r, c]
            tile, _ = env.grid[r][c]

            # Map row index (r) to plot index (plot_r) for the reversed heatmap
            plot_r = grid_size - 1 - r

            x, y = c + 0.5, plot_r + 0.5 # Center of the cell

            text_color = 'black'

            # 1. Handle Terminal States and Barriers
            if tile == 'F':
                ax.text(x, y, 'GOAL', ha='center', va='center', color='black',
                        fontsize=10, fontweight='heavy',
                        bbox={'facecolor': 'yellow', 'alpha': 0.6, 'pad': 1})
                continue
            elif tile == 'B':
                ax.text(x, y, 'B', ha='center', va='center', color='white',
                        fontsize=10, fontweight='heavy',
                        bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 1})
                continue
            elif tile == 'E':
                ax.text(x, y, 'E', ha='center', va='center', color='darkred',
                        fontsize=10, fontweight='heavy',
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 1})
            elif tile == 'S':
                # Show Trash Value (Val) instead of arrow if it's a trash spot
                val = env.grid[r][c][1]
                ax.text(x, y, str(val), ha='center', va='center', color='darkgreen',
                        fontsize=9, fontweight='bold')

            # 2. Draw Arrow for optimal action (only for non-terminal/non-barrier states)
            if action in action_to_arrow and tile not in ('F', 'B'):
                arrow = action_to_arrow[action]
                ax.text(x, y + 0.2, arrow, ha='center', va='center', color=text_color,
                        fontsize=18, fontweight='bold')

    plt.title(title, fontsize=16)
    plt.ylabel("Row (r)")
    plt.xlabel("Column (c)")
    plt.show()
if __name__ == "__main__":
    FPS = 3
    env = MySolver()
    state = env.reset()
    THRESHOLDS = [5, 10, 20, 30, 40]
    MAX_MOVES_PER_EPISODE = 150

    print("Building goal_table...")
    initial_trash_points = env.get_trash_points()
    # reward map for goal_only
    reward_map_goal = build_reward_map(env, is_goal_only=True)
    goal_table, goal_V,goal_convergence_history = env.actions_model(reward_map_goal)
    visualize_convergence(goal_convergence_history)
    visualize_value_heatmap(goal_V, env)
    print("Finished building goal_table.")

    policies_map, v_tables_map = precompute_policies(env, initial_trash_points)

    screen, clock = PygameInit.initialization()
    sum_of_all = 0
    grid_size = env._WallE__grid_size

    for episode in range(5):
        running = True
        current_score = 0
        moves = 0
        state = tuple(env.reset())

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)

            r, c = state
            current_threshold = env.agent_value
            # Mapping to the nearest predefined threshold
            for threshold in THRESHOLDS:
                if current_threshold >= threshold:
                    current_threshold = threshold
                else:
                    break

            # Policy selection based on current waste status and threshold
            current_policy_table,current_v_table = select_policy(env, policies_map,v_tables_map, initial_trash_points,current_threshold)
            # visualize_value_heatmap(current_v_table, env, title=f"Value Function Heatmap ")
            action = int(current_policy_table[r, c])

            moves += 1
            next_state, probability, reward_episode, done = env.step(action, THRESHOLDS)

            # Make sure next_state is a tuple
            if isinstance(next_state, (list, np.ndarray)):
                next_state = tuple(map(int, next_state))
            else:
                next_state = tuple(next_state)

            state = next_state
            current_score += reward_episode

            if done or moves > MAX_MOVES_PER_EPISODE:
                if moves > MAX_MOVES_PER_EPISODE:
                    print(moves)
                print(f"episode {episode} current_score: {current_score}")
                print(f"agent_value at end: {env.agent_value}")
                sum_of_all += current_score
                break

            pygame.display.flip()
            clock.tick(FPS)

    avg = sum_of_all / 5.0
    print(f"Average score: {avg}")
    pygame.quit()