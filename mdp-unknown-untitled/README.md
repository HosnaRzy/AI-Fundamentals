# Fundamentals of Artificial Intelligence - Phase 2 Project

This repository contains the second phase of the AI foundations course, focusing on Markov Decision Processes (MDP) and Reinforcement Learning in a stochastic environment.

## Project Overview

### 1. **WALL-E Environment - Markov Decision Process (MDP)**
**Objective**: Implement an optimal policy for WALL-E to navigate a stochastic grid environment, collect trash, avoid enemies, and ultimately find the plant while managing limited energy and movement constraints.

**Key Components**:
- **Environment**: 8×8 grid with stochastic movement transitions
- **Agent**: WALL-E with power levels [5, 10, 20, 30, 40]
- **Elements**:
  - Buildings (obstacles)
  - Trash items with different values
  - Enemy robots
  - Goal plant
- **Movement**: Stochastic actions with main action and two neighbor actions
- **Power Management**: Agent power affects trash collection efficiency

**Implementation Tasks**:
- Value Iteration algorithm for optimal policy
- Policy precomputation for different meta-states (power levels)
- Heat map visualization for goal-oriented policy
- Convergence analysis with value difference graphs

### 2. **Unknown Environment - Reinforcement Learning**
**Objective**: Implement Q-Learning for WALL-E to learn optimal behavior through trial and error in an unknown environment with the same elements but without prior knowledge of rewards or transition probabilities.

**Key Components**:
- **Learning Algorithm**: Q-Learning with Q-table
- **State Space**: Complex states considering position and agent power
- **Additional Challenge**: Hunter enemy with game-ending penalty
- **Training**: Multiple episodes for policy convergence

**Implementation Tasks**:
- Q-Learning algorithm implementation
- State space definition and Q-table management
- Convergence monitoring with value difference metrics
- Policy map visualization after training

### 3. **Bonus: Deep Q Network (DQN)**
**Objective**: Implement a Deep Q Network as an advanced reinforcement learning approach to handle the same environment using neural network function approximation instead of traditional Q-tables.

**Key Components**:
- **Agent**: WALL-E with power levels [5, 10]
- **Neural Network**: Deep Q-network for Q-value approximation
- **State Representation**: Enhanced state encoding for neural network input

**Implementation Tasks**:
- DQN architecture design and implementation
- Performance comparison with tabular Q-learning

## Learning Objectives

- **Markov Decision Processes**: Understand and implement Value Iteration for stochastic environments
- **Policy Optimization**: Develop comprehensive policies considering multiple state factors
- **Reinforcement Learning**: Apply Q-Learning in unknown environments with exploration-exploitation tradeoffs
- **Deep Reinforcement Learning**: Implement DQN for complex state spaces and compare with traditional methods
- **State Space Management**: Handle complex state representations beyond simple position
- **Performance Evaluation**: Analyze algorithm convergence and agent performance across multiple environments
- **Stochastic Environments**: Work with non-deterministic action outcomes and probabilistic transitions

**Technologies**: Python, PyGame, NumPy, PyTorch/TensorFlow (for DQN)