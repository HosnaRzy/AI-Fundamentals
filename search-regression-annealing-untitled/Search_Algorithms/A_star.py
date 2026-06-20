from env.env import play
from heapq import heappush, heappop
import heapq
import itertools

_CACHED_COST_MATRIX = None

DIRECTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
GRASS_PASSING_COST = 5
BUSH_PASSING_COST = 100
INF = float('inf')

def calculate_terrain_cost_matrix(original_grid):
    rows, cols = original_grid.shape
    cost_matrix = {}

    def get_terrain_cost(r, c):
        if not (0 <= r < rows and 0 <= c < cols):
            return INF

        cell = original_grid[r, c]
        if cell == 'R':
            return INF
        elif cell == 'B':
            return BUSH_PASSING_COST
        else:
            return GRASS_PASSING_COST

    for start_r in range(rows):
        for start_c in range(cols):
            start_pos = (start_r, start_c)

            if original_grid[start_r, start_c] == 'R':
                cost_matrix[start_pos] = {}
                continue

            min_costs = {(r, c): INF for r in range(rows) for c in range(cols)}
            min_costs[start_pos] = 0

            priority_queue = [(0, start_pos)]

            while priority_queue:
                current_cost, (r, c) = heapq.heappop(priority_queue)

                if current_cost > min_costs.get((r, c), INF):
                    continue

                for dr, dc in DIRECTIONS.values():
                    next_r, next_c = r + dr, c + dc
                    next_pos = (next_r, next_c)

                    cost_to_move = get_terrain_cost(next_r, next_c)

                    if cost_to_move != INF:
                        new_cost = current_cost + cost_to_move

                        if new_cost < min_costs.get(next_pos, INF):
                            min_costs[next_pos] = new_cost
                            heapq.heappush(priority_queue, (new_cost, next_pos))

            cost_matrix[start_pos] = min_costs

    return cost_matrix


def calculate_mst_cost(targets_positions, cost_matrix):
    num_targets = len(targets_positions)
    if num_targets <= 1:
        return 0

    min_cost_to_mst = [INF] * num_targets
    in_mst = [False] * num_targets

    min_cost_to_mst[0] = 0
    total_mst_cost = 0

    for _ in range(num_targets):

        min_cost = INF
        min_index = -1

        for i in range(num_targets):
            if not in_mst[i] and min_cost_to_mst[i] < min_cost:
                min_cost = min_cost_to_mst[i]
                min_index = i

        if min_index == -1:
            break

        in_mst[min_index] = True
        total_mst_cost += min_cost
        current_target_pos = targets_positions[min_index]

        for i in range(num_targets):
            if not in_mst[i]:
                next_target_pos = targets_positions[i]

                try:
                    edge_cost = cost_matrix[current_target_pos].get(next_target_pos, INF)
                except KeyError:
                    edge_cost = INF

                if edge_cost < min_cost_to_mst[i]:
                    min_cost_to_mst[i] = edge_cost

    return total_mst_cost

def enemy_penalty(state):
    enemy = state.get_enemy_position()
    agent = state.get_agent_position()
    if not enemy:
        return 0
    dist_enemy = abs(agent[0] - enemy[0]) + abs(agent[1] - enemy[1])
    return dist_enemy

def heuristic(state):
    global _CACHED_COST_MATRIX
    if _CACHED_COST_MATRIX is None:
        _CACHED_COST_MATRIX = calculate_terrain_cost_matrix(state._original_grid)

    cost_matrix = _CACHED_COST_MATRIX
    targets = state.get_targets_positions()
    agent_pos = state.get_agent_position()

    if not targets:
        return 0

    min_cost_to_nearest_target = INF

    try:
        for target_pos in targets:
            cost = cost_matrix[agent_pos].get(target_pos, INF)
            if cost < min_cost_to_nearest_target:
                min_cost_to_nearest_target = cost
    except KeyError:
        return 1000000

    mst_cost = calculate_mst_cost(targets, cost_matrix)

    if min_cost_to_nearest_target == INF:
        return 1000000

    return min_cost_to_nearest_target + mst_cost + enemy_penalty(state)

def a_star(state):
    frontier = []
    counter = itertools.count()
    cost_so_far = {state: 0}

    heappush(frontier, (0, next(counter), state, []))

    while frontier:
        f, seq, state, path = heappop(frontier)
        g = cost_so_far[state]

        if state.is_goal_state():
            return path

        for action, step_cost, next_state in state.get_successors(toward_walls=True):
            if next_state.is_collision_state():
                continue

            g_new = g + step_cost
            h = heuristic(next_state)
            w = 3
            f_new = g_new + (w*h)

            if next_state not in cost_so_far or g_new < cost_so_far[next_state]:
                cost_so_far[next_state] = g_new
                heappush(frontier, (f_new, next(counter), next_state, path + [action]))
    return None

if __name__ == "__main__":
    play('medium', a_star, delay=500)