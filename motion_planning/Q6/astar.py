import numpy as np
import matplotlib.pyplot as plt
import heapq

def initialise_exploration_grid(grid_size, obstacles):
    """
    Initializes the "exploration grid"
    This is a grid of 0s for unexplored free space, and 1s for explored space or obstacles
    Obstacles are rectangles defined by their extreme coordinates: (x_min, y_min, x_max, y_max)
    Each obstacle i unpacked from the list of obstacles
    """
    exploration_grid = np.ones((grid_size, grid_size))

    for obstacle in obstacles:
        x_min, y_min, x_max, y_max = obstacle
        exploration_grid[x_min : x_max + 1, y_min : y_max + 1] = 0

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
    """
    Calculates the cost to go from the specified vertex to the goal by one of three heuristics.
    "zero" - reduces A* to Djikstra's algorithm
    "euclidean" - euclidean distance to the goal
    "inflated" - euclidean multiplied by some value (used here is 100) to bias movement towards the goal
    """
    if heuristic == "zero":
        cost_to_go = 0
    elif heuristic == "euclidean":
        cost_to_go = np.linalg.norm(np.array(vertex) - np.array(goal))
    elif heuristic == "inflated":
        cost_to_go = 100 * np.linalg.norm(np.array(vertex) - np.array(goal))
    else:
        raise ValueError("Incompatible Heuristic!1")

    return cost_to_go

def astar(start, goal, grid_size, obstacles, heuristic):

    cost_grid = initialise_cost_grid(grid_size, start)
    exploration_grid = initialise_exploration_grid(grid_size, obstacles)
    
    # Priority queue storing (total_cost, (x, y))
    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from = {}
    expanded_vertices = 0
    all_paths = []

    # While there are accessible nodes left to explore
    while frontier:

         # Pop node with lowest cost from the heap
        _, current_vertex = heapq.heappop(frontier)

        if current_vertex == goal:
            break

        cost_to_come = cost_grid[current_vertex[0], current_vertex[1]]

        # Cost to come for all neighbours is the same since all edges are assumed to have weight 1
        neighbour_cost_to_come = cost_to_come + 1
        neighbours = get_neighbors_coordinates(current_vertex, exploration_grid)

        for neighbour in neighbours:

            neighbour_x, neighbour_y = neighbour
            neighbour_cost_to_go = cost_to_go(heuristic, neighbour, goal)
            neighbour_total_cost = neighbour_cost_to_come + neighbour_cost_to_go

            if neighbour_total_cost < cost_grid[neighbour_x, neighbour_y]:
                cost_grid[neighbour_x, neighbour_y] = neighbour_total_cost
                came_from[neighbour] = current_vertex
                heapq.heappush(frontier, (neighbour_total_cost, neighbour))
                all_paths.append((current_vertex, neighbour))
        
        exploration_grid[current_vertex[0], current_vertex[1]] = 0

        masked_cost_grid = np.where(exploration_grid == 1, cost_grid, np.inf)
        current_vertex = np.unravel_index(np.argmin(masked_cost_grid), masked_cost_grid.shape)
        expanded_vertices += 1

    final_path= []
    if current_vertex == goal:
        while current_vertex in came_from:
            final_path.append(current_vertex)
            current_vertex = came_from[current_vertex]
        final_path.append(start)
        final_path.reverse()
        total_cost = len(final_path) - 1
    else:
        print("Could not find a solution!")
        total_cost = "No path found!"

    return final_path, all_paths, exploration_grid, expanded_vertices, total_cost

def visualize_astar(grid_size, start, goal, obstacles, final_path, all_paths, expanded_vertices, total_cost):

    _, ax = plt.subplots(figsize=(8, 8))

    # Create grid and mark obstacle locations
    grid = np.zeros((grid_size, grid_size))
    for obstacle in obstacles:
        x_min, y_min, x_max, y_max = obstacle
        grid[x_min : x_max + 1, y_min : y_max + 1] = 1
    ax.imshow(grid.T, cmap="gray_r", origin="lower")

    # Plot all explored paths
    for edge in all_paths:
        node_1, node_2 = edge
        ax.plot([node_1[0], node_2[0]], [node_1[1], node_2[1]], color="orange", linewidth=1)
    ax.plot([], [], color="orange", linewidth=1, label="Explored Paths")

    # Plot the final path
    if final_path:
        px, py = zip(*final_path)
        ax.plot(px, py, marker="o", color="blue", markersize=5, label="Final Path")

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
            fontsize=12, verticalalignment='top', horizontalalignment='left', color='black')
    ax.text(0.05, 0.90, f"Total Cost: {total_cost}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left', color='black')
    plt.show()

# Remember to subtract 1 from all coordinates as indexing starts from 0
grid_size = 19
start = (4, 9)
goal = (14, 9)
# obstacles = [(9, 4, 9, 14)] # (x_min, y_min, x_max, y_max)
obstacles = [(8, 7, 8, 11), (2, 6, 8, 6), (2, 12, 8, 12)]
heuristic = "euclidean" # "zero", "euclidean", or "inflated"

final_path, all_paths, exploration_grid, expanded_vertices, total_cost = astar(start, goal, grid_size, obstacles, heuristic)
visualize_astar(grid_size, start, goal, obstacles, final_path, all_paths, expanded_vertices, total_cost)