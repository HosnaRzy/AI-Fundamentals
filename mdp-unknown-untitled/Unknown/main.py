import numpy as np
import pygame
import os
from collections import defaultdict
from environment import WallE_RL, PygameInit, ACTIONS
import matplotlib.pyplot as plt
import seaborn as sns 

# -------------------- Hyperparameters --------------------
TRAIN_EPISODES = 7000
MAX_ACTIONS_PER_EPISODE = 200
NUM_TEST_EPISODES = 5
THRESHOLDS = [5, 10, 20, 30]
ALPHA = 0.1                  # learning rate
GAMMA = 0.999                  # discount factor
EPS_START = 1.0              # initial epsilon (exploration)
EPS_END = 0.05               # minimum epsilon
EPS_DECAY = 0.99995           # decay rate
CONVERGENCE_THRESHOLD = 0.01
# ---------------------------------------------------------

def discretize_power(agent_power, thresholds=THRESHOLDS):  
    """
    Return the index of the largest threshold <= agent_power.    
    """      
    idx = 0
    for i, t in enumerate(thresholds):
        if agent_power >= t:
            idx = i
        else:
            break
    return idx

def state_to_key(state, agent_power, trash_state_tuple):
    """
    Output: a tuple key for the Q-table: (r, c, power_bucket, *trash_tuple)    
    """  
    r, c = state
    p_bucket = discretize_power(agent_power)
    return (int(r), int(c), int(p_bucket)) + tuple(trash_state_tuple)

def epsilon_greedy_action(q_table, s_key, epsilon):
    """Action selection with epsilon-greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)    
    q_vals = q_table[s_key]    
    max_a = int(np.argmax(q_vals))
    return max_a

def compute_value_difference(old_q_serial, q_table):
    """
    Compute the Value Difference metric:   
    """
    max_change = 0.0
    # union of keys
    keys = set(old_q_serial.keys()) | set(q_table.keys())
    for k in keys:
        old = np.array(old_q_serial.get(k, np.zeros(len(ACTIONS))), dtype=float)
        new = np.array(q_table.get(k, np.zeros(len(ACTIONS))), dtype=float)
        diff = np.abs(new - old)
        cur_sum = np.sum(diff)
        max_change += cur_sum
    return float(max_change)

def train(env, q_table, num_episodes=TRAIN_EPISODES):
    epsilon = EPS_START
    episode_rewards = []   
    value_diffs = []     

    for ep in range(1, num_episodes + 1): 
        old_q_serial = {k: np.array(v, dtype=float) for k, v in q_table.items()}       
        state, agent_power = env.reset()
        current_trash_state = tuple(env._WallE_RL__get_trash_state())
        s_key = state_to_key(state, agent_power, current_trash_state)
        total_reward = 0.0

        for step_idx in range(MAX_ACTIONS_PER_EPISODE):
            a = epsilon_greedy_action(q_table, s_key, epsilon)
            
            next_state, reward, next_trash_state, done, next_agent_power = env.step(a)

            total_reward += reward

            next_s_key = state_to_key(next_state, next_agent_power, tuple(next_trash_state))
        
            _ = q_table[s_key] 
            _ = q_table[next_s_key]

            q_sa = q_table[s_key][a]
            q_next_max = np.max(q_table[next_s_key])

            # Q-learning update rule
            q_table[s_key][a] = q_sa + ALPHA * (reward + GAMMA * q_next_max - q_sa)

            s_key = next_s_key
            current_trash_state = next_trash_state

            if done:
                break
            
            epsilon = max(EPS_END, epsilon * EPS_DECAY)                       

        episode_rewards.append(total_reward)   
        vd = compute_value_difference(old_q_serial, q_table)
        value_diffs.append(vd)    
    
        if ep >= 1000 and vd < CONVERGENCE_THRESHOLD:
            print("-" * 60)            
            print("converged")
            print("-" * 60)
            break 
        
        # periodic logging
        if ep % 100 == 0 or ep == 1:
            avg_recent = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 1 else 0.0
            print(f"[TRAIN] Ep {ep}/{num_episodes}  reward={total_reward:.1f}  avg100={avg_recent:.2f}  eps={epsilon:.4f}")

    return q_table, episode_rewards, value_diffs

def extract_policy_map_from_qtable(q_table, grid_size, power_bucket, initial_trash_state):
    """
    Extracts the optimal policy (best action) for every cell (r, c)    
    """    
    policy_table = np.full((grid_size, grid_size), -1, dtype=np.int8)
        
    base_key_suffix = (power_bucket,) + tuple(initial_trash_state)
    
    for r in range(grid_size):
        for c in range(grid_size):
            s_key = (r, c) + base_key_suffix
                        
            if s_key in q_table:                
                optimal_action = np.argmax(q_table[s_key])
                policy_table[r, c] = optimal_action
            else:                
                policy_table[r, c] = -1 
                
    return policy_table

def visualize_policy_arrows(policy_table, env, title="Optimal Policy Map"):
    """
    Visualizes the 8x8 policy table using arrows on a grayscale grid.
    """
    grid_size = policy_table.shape[0]        
    policy_reversed = policy_table[::-1] 
    plt.figure(figsize=(8, 8))        
    sns.heatmap(
        np.zeros((grid_size, grid_size)), 
        cmap="Greys", 
        linewidths=.5, 
        linecolor='black',
        cbar=False,
        annot=False,
        xticklabels=True,
        yticklabels=True
    )    
    action_to_arrow = {0: '↑', 1: '↓', 2: '←', 3: '→', -1: '•'}

    for r in range(grid_size):
        for c in range(grid_size):            
            action = policy_reversed[r, c]                    
            r_orig = grid_size - 1 - r 
            tile, val = env._WallE_RL__grid[r_orig][c]

            x, y = c + 0.5, r + 0.5
            text_color = 'black'            
            if tile == 'F':
                display_text = 'GOAL'
                bbox_params = {'facecolor': 'yellow', 'alpha': 0.7, 'pad': 2}
            elif tile == 'B':
                display_text = 'B'
                text_color = 'white'
                bbox_params = {'facecolor': 'red', 'alpha': 0.8, 'pad': 2}
            elif tile == 'E':
                display_text = 'E'
                text_color = 'darkred'
                bbox_params = {'facecolor': 'white', 'alpha': 0.8, 'pad': 1}
            elif tile == 'S':                
                display_text = str(val)
                text_color = 'darkgreen'
                bbox_params = {}
            else:
                display_text = action_to_arrow.get(action, '')
                text_color = 'black'
                bbox_params = {}            
            ax = plt.gca()
            ax.text(x, y, display_text, ha='center', va='center', color=text_color,
                    fontsize=16 if tile not in ['S', 'F', 'B', 'E'] else 10, 
                    fontweight='bold',
                    bbox=bbox_params)

    plt.title(title, fontsize=16)
    plt.ylabel("Row (r)")
    plt.xlabel("Column (c)")
    plt.show()    

if __name__ == "__main__":

    FPS = 2
    env = WallE_RL()        
    print("Environment initialized. Starting training...")
    
    n_actions = len(ACTIONS)  
    q_table = defaultdict(lambda: np.zeros(n_actions, dtype=float))

    # --- Train ---
    q_table, train_rewards, value_diffs = train(env, q_table, num_episodes=TRAIN_EPISODES)
    episodes = range(1, len(train_rewards) + 1)
    window_size = 50 
      
    # ----------------- 1. Value Difference Plot (نمودار همگرایی) -----------------
    plt.figure(figsize=(12, 5))
    
    plt.plot(episodes, value_diffs, label='Value Difference (Raw)', color='blue', alpha=0.3)
    
    if len(value_diffs) >= window_size:
        moving_avg_vd = np.convolve(value_diffs, np.ones(window_size)/window_size, mode='valid')        
        plt.plot(episodes[window_size - 1:], moving_avg_vd, label=f'VD Moving Average ({window_size} Eps)', color='red', linewidth=3)
        
    plt.axhline(y=CONVERGENCE_THRESHOLD, color='gray', linestyle='--', label=f'Convergence Threshold ({CONVERGENCE_THRESHOLD})')
    
    plt.title('Convergence Analysis: Value Difference vs. Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Value Difference: Sum(|Q_k - Q_{k-1}|)')
    plt.yscale('log')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()    

    TARGET_POWER_BUCKET = 5    
    INITIAL_TRASH_STATE = tuple(env._WallE_RL__get_trash_state()) 
    grid_size = env._WallE_RL__grid_size    
    # 2. Extract the Policy Map
    policy_map = extract_policy_map_from_qtable(
        q_table, 
        grid_size, 
        TARGET_POWER_BUCKET, 
        INITIAL_TRASH_STATE
    )
    
    # 3. Visualize the Policy Map
    visualize_policy_arrows(
        policy_map, 
        env, 
        title=f"Optimal Policy Map (Q-Learning) \n Power Bucket: {TARGET_POWER_BUCKET}, All Trash Present"
    )

    # --- Test ---
    print(f"Training finished. Starting test run for {NUM_TEST_EPISODES} episodes...")
    screen, clock = PygameInit.initialization()  
    episode_rewards = []

    # Main game loop
    for episode in range(NUM_TEST_EPISODES):
        state, agent_power = env.reset()    
        current_trash_state = tuple(env._WallE_RL__get_trash_state())
        s_key = state_to_key(state, agent_power, current_trash_state)    
        total_reward = 0.0      
        steps = 0
        done = False    
      
        while not done and steps < MAX_ACTIONS_PER_EPISODE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            env.render(screen)   
            
            _ = q_table[s_key] 
            a = int(np.argmax(q_table[s_key]))

            next_state, reward, next_trash_state, done, next_agent_power = env.step(a)
            
            total_reward += reward
            s_key = state_to_key(next_state, next_agent_power, tuple(next_trash_state))
            steps += 1

            pygame.display.flip()
            clock.tick(FPS)
          

        episode_rewards.append(total_reward)
        print(f"[TEST] Episode {episode+1} reward = {total_reward:.1f}")                 
    
    print("-" * 40)
    print(f"MEAN REWARD (TRAIN avg last 100): {np.mean(train_rewards[-100:]) if len(train_rewards)>=1 else np.mean(train_rewards)}")
    print(f"MEAN REWARD (TEST): {np.mean(episode_rewards)}")    
    pygame.quit()