import numpy as np

def initialise_exploration_grid(grid_size, obstacles):
    """
    Returns a square matrix of side length grid_size
    0s represent obstacles and explored (closed) nodes (initially none)
    1s represent unexplored (open) nodes
    """
    exploration_grid = np.ones((grid_size, grid_size))
    for obstacle in obstacles:
        exploration_grid[obstacle[0], obstacle[1]] = 0

    return exploration_grid

def initialise_cost_grid(grid_size, start):
    """
    Returns a square matrix side length grid_size
    Initialises all values to infinity apart from the start position
    """
    cost_grid = np.ones((grid_size, grid_size)) * np.inf
    cost_grid[start[0], start[1]] = 0

    return cost_grid

def get_neighbors_coordinates(current_vertex, exploration_grid):
    """
    Returns a list of coordinates of all open neighbours
    """
    x, y = current_vertex
    neighbors_coords = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] # 8-connected i.e can move diagonally
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < exploration_grid.shape[0] and 0 <= ny < exploration_grid.shape[1] and exploration_grid[nx, ny] == 1:
            neighbors_coords.append((nx, ny))

    return neighbors_coords

def cost_to_come(cost_grid, vertex):
    # REDUNDANT
    cost_to_come = cost_grid[vertex[0], vertex[1]] + 1 # Only exploring nextdoor neighbour hence +1 distance always

    return cost_to_come

def astar(start, goal, grid_size, obstacles):

    cost_grid = initialise_cost_grid(grid_size, start)
    exploration_grid = initialise_exploration_grid(grid_size, obstacles)

    current_vertex = start

    while tuple(current_vertex) != tuple(goal) and np.any(exploration_grid == 1):

        cost_to_come = cost_grid[current_vertex[0], current_vertex[1]]
        neighbour_cost_to_come = cost_to_come + 1
        neighbours = get_neighbors_coordinates(current_vertex, exploration_grid)

        for neighbour in neighbours:
            if neighbour_cost_to_come < cost_grid[neighbour[0], neighbour[1]]:
                cost_grid[neighbour[0], neighbour[1]] = neighbour_cost_to_come
        
        exploration_grid[current_vertex[0], current_vertex[1]] = 0

        masked_cost_grid = np.where(exploration_grid == 1, cost_grid, np.inf)
        current_vertex = np.unravel_index(np.argmin(masked_cost_grid), masked_cost_grid.shape)

    if current_vertex[0] == goal[0] and current_vertex[1] == goal[1]:
        print("Made it!")
    else:
        print("Error!")

grid_size = 19
start = np.array((5, 10))
goal = np.array((15, 10))
obstacles = [(10, 5),(10, 6)]
astar(start, goal, grid_size, obstacles)