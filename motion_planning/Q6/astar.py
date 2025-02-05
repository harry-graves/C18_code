import numpy as np
import matplotlib.pyplot as plt

def initialise_exploration_grid(grid_size, obstacles):
    """
    Initializes the exploration grid with obstacles.
    Obstacles can be defined as single points or lines between two points.
    """
    exploration_grid = np.ones((grid_size, grid_size))

    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle  # Extract line endpoints
        if x1 == x2:  # Vertical line
            exploration_grid[x1, min(y1, y2):max(y1, y2) + 1] = 0
        elif y1 == y2:  # Horizontal line
            exploration_grid[min(x1, x2):max(x1, x2) + 1, y1] = 0
        else:
            raise ValueError("Only horizontal or vertical obstacles are supported.")

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
    came_from = {}

    while tuple(current_vertex) != tuple(goal) and np.any(exploration_grid == 1):

        cost_to_come = cost_grid[current_vertex[0], current_vertex[1]]
        neighbour_cost_to_come = cost_to_come + 1
        neighbours = get_neighbors_coordinates(current_vertex, exploration_grid)

        for neighbour in neighbours:
            if neighbour_cost_to_come < cost_grid[neighbour[0], neighbour[1]]:
                cost_grid[neighbour[0], neighbour[1]] = neighbour_cost_to_come
                came_from[tuple(neighbour)] = tuple(current_vertex)
        
        exploration_grid[current_vertex[0], current_vertex[1]] = 0

        masked_cost_grid = np.where(exploration_grid == 1, cost_grid, np.inf)
        current_vertex = np.unravel_index(np.argmin(masked_cost_grid), masked_cost_grid.shape)

    path = []
    if current_vertex[0] == goal[0] and current_vertex[1] == goal[1]:
        print("Made it!")
        while current_vertex in came_from:
            path.append(current_vertex)
            current_vertex = came_from[current_vertex]
        path.append(tuple(start))
        path.reverse()
    else:
        print("Error!")

    return path, exploration_grid

def visualize(grid_size, start, goal, obstacles, path):
    """Visualizes the grid, obstacles, and the found path using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create empty grid
    grid = np.ones((grid_size, grid_size))

    # Mark obstacles
    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle
        if x1 == x2:  # Vertical wall
            grid[x1, min(y1, y2):max(y1, y2) + 1] = 0
        elif y1 == y2:  # Horizontal wall
            grid[min(x1, x2):max(x1, x2) + 1, y1] = 0

    ax.imshow(grid.T, cmap="gray_r", origin="lower")

    # Plot path
    if path:
        px, py = zip(*path)
        ax.plot(px, py, marker="o", color="blue", markersize=5, label="Path")

    # Mark start and goal
    ax.scatter(*start, color="green", s=100, label="Start", edgecolors="black")
    ax.scatter(*goal, color="red", s=100, label="Goal", edgecolors="black")

    # Show plot
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.show()

grid_size = 19
start = np.array((5, 10))
goal = np.array((15, 10))
obstacles = [(10, 5, 10, 15)]

path, exploration_grid = astar(start, goal, grid_size, obstacles)
visualize(grid_size, start, goal, obstacles, path)