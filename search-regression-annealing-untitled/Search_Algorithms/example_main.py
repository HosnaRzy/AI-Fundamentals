from env.env import play

from collections import deque

def bfs(state):
    frontier = deque([(state, [])])
    explored = set()

    while frontier:
        state, actions = frontier.popleft()

        if state.is_goal_state():
            return actions

        if state not in explored:
            explored.add(state)
            for action, cost, next_state in state.get_successors():
                if next_state.is_collision_state():
                    continue
                if next_state not in explored:
                    frontier.append((next_state, actions + [action]))

    return []
    # Implement Search algorithm here


if __name__ == "__main__":
    play('easy', bfs, delay=500)