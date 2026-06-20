from env.env import play
from heapq import heappush, heappop
import itertools
def ucs(state):
    frontier = []
    counter = itertools.count()
    cost_so_far = {state: 0}

    heappush(frontier, (0,next(counter),state,[])) # item: cost, state, path

    while frontier:
        cost, seq, state, path = heappop(frontier)
        if state.is_goal_state():
            return path

        if cost > cost_so_far[state]:
            continue

        for action, step_cost, next_state in state.get_successors():
            if next_state.is_collision_state():
                continue

            new_cost = cost + step_cost

            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                heappush(frontier, (new_cost, next(counter), next_state, path + [action]))
    return None

if __name__ == "__main__":
    play('medium', ucs, delay=500)