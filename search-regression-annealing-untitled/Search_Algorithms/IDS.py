from env.env import play

CUT_OFF = "cutoff"
def dls(initial_state, limit):
    visited = set()
    frontier = [(initial_state, [], frozenset([initial_state]))]
    result = None
    while frontier:
        state, path, ancestors = frontier.pop()
        if state.is_goal_state():
            return path

        depth = len(path)
        #depth more than limit
        if depth > limit:
            result = CUT_OFF
            continue
        try:
            successors = state.get_successors()
        except TypeError:
            successors = state.get_successors(toward_walls=False)

        for action, step_cost, next_state in successors:
            #repeated state
            if next_state in visited:
                continue
            visited.add(next_state)
            #enemy collision
            if next_state.is_collision_state():
                continue
            #creating cycle
            if next_state in ancestors:
                continue
            child_path = path + [action]
            #cutoff happening
            if len(child_path) > limit:
                result = CUT_OFF
                continue

            new_ancestors = set(ancestors)
            new_ancestors.add(next_state)
            frontier.append((next_state, child_path, frozenset(new_ancestors)))
    return result


def ids(initial_state, start_depth=64, max_depth=100):
    for depth in range(start_depth, max_depth + 1):
        result = dls(initial_state, depth)
        if result != CUT_OFF and result is not None:
            return result or []
    return []

if __name__ == "__main__":
    play('medium', ids, delay=500)