def _grid_search_iter(grid, keys, values, index):
    if index == len(grid):
        return

    key, param_values = keys[index], values[index]
    base_iter = _grid_search_iter(grid, keys, values, index + 1)
    for base_params in base_iter:
        for param_value in param_values:
            params = base_params.copy()
            params[key] = param_value
            yield params

def grid_search_iter(grid):
    return _grid_search_iter(grid, grid.keys(), grid.values(), index=0)
