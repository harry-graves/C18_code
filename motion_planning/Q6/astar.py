import numpy as np
import matplotlib.pyplot as plt
import heapq

def initialise_exploration_grid(grid_size, obstacles):
    """
    Initializes the exploration grid with obstacles.
    Obstacles can be defined as single points or lines between two points.
    Obstacle is defined as (x1, y1, x2, y2) i.e. the start and end coordinates of the obstacle
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

def cost_to_go(heuristic, vertex, goal):

    if heuristic == "zero":
        cost_to_go = 0
    elif heuristic == "euclidean":
        cost_to_go = np.linalg.norm(vertex - goal)
    elif heuristic == "inflated":
        cost_to_go = 100 * np.linalg.norm(vertex - goal)
    else:
        raise ValueError("Incompatible Heuristic!1")

    return cost_to_go

def astar(start, goal, grid_size, obstacles, heuristic):

    cost_grid = initialise_cost_grid(grid_size, start)
    exploration_grid = initialise_exploration_grid(grid_size, obstacles)
    
    # Priority queue storing (total_cost, (x, y))
    frontier = []
    heapq.heappush(frontier, (0, tuple(start)))

    came_from = {}
    expanded_vertices = 0

    while frontier:
        _, current_vertex = heapq.heappop(frontier) # Pop node with lowest cost from the heap

        if current_vertex == tuple(goal):
            break

        cost_to_come = cost_grid[current_vertex[0], current_vertex[1]]
        neighbour_cost_to_come = cost_to_come + 1
        neighbours = get_neighbors_coordinates(current_vertex, exploration_grid)

        for neighbour in neighbours:

            neighbour_cost_to_go = cost_to_go(heuristic, neighbour, goal)
            neighbour_total_cost = neighbour_cost_to_come + neighbour_cost_to_go

            if neighbour_total_cost < cost_grid[neighbour[0], neighbour[1]]:
                cost_grid[neighbour[0], neighbour[1]] = neighbour_total_cost
                came_from[tuple(neighbour)] = tuple(current_vertex)
                heapq.heappush(frontier, (neighbour_total_cost, neighbour))                
        
        exploration_grid[current_vertex[0], current_vertex[1]] = 0

        masked_cost_grid = np.where(exploration_grid == 1, cost_grid, np.inf)
        current_vertex = np.unravel_index(np.argmin(masked_cost_grid), masked_cost_grid.shape)
        expanded_vertices += 1

    path = []
    if current_vertex == tuple(goal):
        while current_vertex in came_from:
            path.append(current_vertex)
            current_vertex = came_from[current_vertex]
        path.append(tuple(start))
        path.reverse()
        total_cost = len(path) - 1
    else:
        print("Could not find a solution!")
        total_cost = "No path found!"

    return path, exploration_grid, expanded_vertices, total_cost

def visualize_astar(grid_size, start, goal, obstacles, path, expanded_vertices, total_cost):
    """Visualizes the grid, obstacles, and the found path using Matplotlib."""
    _, ax = plt.subplots(figsize=(8, 8))

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
    ax.set_title(f"Path using {heuristic} heuristic")
    ax.text(0.05, 0.95, f"Expanded Vertices: {expanded_vertices}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left', color='white')
    ax.text(0.05, 0.90, f"Total Cost: {total_cost}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left', color='white')
    plt.show()

# Remember to subtract 1 from all coordinates as indexing starts from 0
grid_size = 19
start = np.array((4, 9))
goal = np.array((14, 9))
obstacles = [(8, 7, 8, 11), (2, 6, 8, 6), (2, 12, 8, 12)]
heuristic = "inflated" # "zero", "euclidean", or "inflated"

path, exploration_grid, expanded_vertices, total_cost = astar(start, goal, grid_size, obstacles, heuristic)
visualize_astar(grid_size, start, goal, obstacles, path, expanded_vertices, total_cost)