import numpy as np
import matplotlib.pyplot as plt

def add_sample(grid_size):

    x = np.random.random_sample() * grid_size
    y = np.random.random_sample() * grid_size

    return tuple((x, y))

def add_new_node(nearest_node, sample, max_distance):

    direction = np.array(sample) - np.array(nearest_node)
    length = np.linalg.norm(direction)
    if length < max_distance:
        return tuple(sample)
    direction = (direction / length) * max_distance
    new_node = np.array(nearest_node) + direction

    return tuple(new_node)

def node_collision_check(sample, obstacles):

    for (x_min, y_min, x_max, y_max) in obstacles:
        if x_min <= sample[0] <= x_max and y_min <= sample[1] <= y_max:
            return True
        
    return False

def find_nearest_node(nodes, sample, grid_size):

    distance = grid_size
    nearest_node = nodes[0]
    for node in nodes:
        new_distance = np.linalg.norm(np.array(node) - np.array(sample))
        if new_distance < distance:
            distance = new_distance
            nearest_node = node

    return nearest_node

def edge_collision_check(vertex_1, vertex_2, obstacles, step_size = 0.1):
    """
    Checks if the edge from start to end collides with any obstacle.
    Uses small step increments to sample along the line.
    """
    x1, y1 = vertex_1
    x2, y2 = vertex_2

    distance = np.linalg.norm(np.array(vertex_2) - np.array(vertex_1))
    if distance == 0:
        return False

    num_steps = max(int(distance / step_size), 1)

    for i in range(num_steps + 1):
        # Interpolate point along the line
        alpha = i / num_steps
        x = (1 - alpha) * x1 + alpha * x2
        y = (1 - alpha) * y1 + alpha * y2

        for (x_min, y_min, x_max, y_max) in obstacles:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True

    return False

def calculate_path_length(path):

    path_length = 0

    if path is None:
        path_length = None
    else:
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))

    return path_length

def rrt(grid_size, start, goal, obstacles, max_distance, edge_collision_check_size, max_iterations):

    nodes = [start]
    distance_to_goal = np.linalg.norm(np.array(start) - np.array(goal))

    if distance_to_goal < max_distance:
        return [start, goal], [start, goal], {goal: start}

    i = 0
    came_from = {start: None}

    while distance_to_goal > max_distance and i <= max_iterations:

        sample = add_sample(grid_size)
        collision = node_collision_check(sample, obstacles)
        if collision == True:
            continue

        nearest_node = find_nearest_node(nodes, sample, grid_size)
        new_node = add_new_node(nearest_node, sample, max_distance)
        collision = node_collision_check(new_node, obstacles)
        if collision == True:
            continue

        collision = edge_collision_check(new_node, nearest_node, obstacles, edge_collision_check_size)
        if collision == True:
            continue

        nodes.append(new_node)
        came_from[new_node] = nearest_node

        distance_to_goal = np.linalg.norm(np.array(new_node) - np.array(goal))
        if distance_to_goal < max_distance:
            nodes.append(goal)
            came_from[goal] = new_node
            break

        i+=1

    path = [goal]
    if goal in came_from:
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()

    else:
        path = None
        print("Could not find a path!")
    
    return nodes, path, came_from

def visualize_rrt(grid_size, start, goal, obstacles, nodes, path, came_from):
    """
    Visualizes the RRT tree, obstacles, and the final path.
    """
    _, ax = plt.subplots(figsize=(8, 8))
    
    for (x_min, y_min, x_max, y_max) in obstacles:
        width = x_max - x_min
        height = y_max - y_min
        ax.add_patch(plt.Rectangle((x_min, y_min), width, height, color='gray'))

    # Plot all RRT nodes
    for node in nodes:
        ax.scatter(node[0], node[1], color='blue', s=5)  # Small blue dots

    # Plot edges (tree structure)
    for child, parent in came_from.items():
        if parent is not None:
            plt.plot([parent[0], child[0]], [parent[1], child[1]], color="blue", linewidth=0.5)

    # Plot path if found
    if path:
        px, py = zip(*path)
        ax.plot(px, py, color="red", linewidth=2, marker="o", markersize=5, label="Path")

    # Mark start and goal
    ax.scatter(*start, color="green", s=100, label="Start", edgecolors="black")
    ax.scatter(*goal, color="red", s=100, label="Goal", edgecolors="black")

    # Performance metrics
    path_length = calculate_path_length(path)
    text_x, text_y = 1, grid_size - 1  # Position for text
    ax.text(text_x, text_y, f"Nodes Expanded: {len(nodes)}", fontsize=12, color="black")
    if path_length is not None:
        ax.text(text_x, text_y - 1, f"Path Length: {path_length:.2f}", fontsize=12, color="black")
    else:
        ax.text(text_x, text_y - 1, f"Path Length: No path found!", fontsize=12, color="black")

    # Formatting
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("RRT Pathfinding")
    ax.legend()
    plt.show()

grid_size = 19
start = tuple((4.5, 9.5))
goal = tuple((14.5, 9.5))
# obstacles = [(9, 1, 10, 18)]
obstacles = [(2, 6, 8, 7), (2, 12, 8, 13), (7, 7, 8, 12)] # (xmin, ymin, xmax, ymax)
max_distance = 1
edge_collision_check_size = 0.05
max_iterations = 1000

nodes, path, came_from = rrt(grid_size, start, goal, obstacles, max_distance, edge_collision_check_size, max_iterations)
visualize_rrt(grid_size, start, goal, obstacles, nodes, path, came_from)