from rrt import add_sample, node_collision_check, edge_collision_check, calculate_path_length
import numpy as np
import matplotlib.pyplot as plt
import heapq

def dijkstra(start, goal, nodes):
    queue = [(0, start)]
    visited = set()
    came_from = {start: None}
    cost_so_far = {start: 0}

    while queue:
        cost, current = heapq.heappop(queue)
        if current == goal:
            break

        if current in visited:
            continue
        visited.add(current)

        for neighbor in nodes.get(current, []):
            new_cost = cost_so_far[current] + np.linalg.norm(np.array(neighbor) - np.array(current))
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    if goal not in came_from:
        return None  # No path found
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

def find_nearby_nodes(sample, nodes, distance_threshold):

    nearby_nodes = []
    for node in nodes.keys():

        distance = np.linalg.norm(np.array(node) - np.array(sample))
        if distance <= distance_threshold:
            nearby_nodes.append(node)

    return nearby_nodes

def preprocess(grid_size, num_samples, obstacles, distance_threshold, edge_collision_check_size):

    nodes = {}
    i = 0
    while i < num_samples:
        
        sample = add_sample(grid_size)
        collision = node_collision_check(sample, obstacles)
        if collision == True:
            continue

        nodes.setdefault(sample, [])

        nearby_nodes = find_nearby_nodes(sample, nodes, distance_threshold)

        for neighbour in nearby_nodes:
            collision = edge_collision_check(sample, neighbour, obstacles, edge_collision_check_size)
            if collision == True:
                continue

            nodes[sample].append(neighbour)
            nodes[neighbour].append(sample)

        i += 1

    return nodes

def query(start, goal, nodes, obstacles, distance_threshold, edge_collision_check_size):

    for sample in [start, goal]:

        nodes.setdefault(sample, [])

        nearby_nodes = find_nearby_nodes(sample, nodes, distance_threshold)

        for neighbour in nearby_nodes:
            collision = edge_collision_check(sample, neighbour, obstacles, edge_collision_check_size)
            if collision == True:
                continue

            nodes[sample].append(neighbour)
            nodes[neighbour].append(sample)

    return nodes

def visualize_prm(nodes, grid_size, obstacles, start, goal, path):

    _, ax = plt.subplots(figsize=(8, 8))

    # Plot edges
    for node, neighbors in nodes.items():
        for neighbor in neighbors:
            ax.plot([node[0], neighbor[0]], [node[1], neighbor[1]], color='blue', linewidth=0.5)

    # Plot nodes
    for node in nodes:
        ax.scatter(*node, color='blue', s=5)

    # Plot obstacles
    for x_min, y_min, x_max, y_max in obstacles:
        width = x_max - x_min
        height = y_max - y_min
        ax.add_patch(plt.Rectangle((x_min, y_min), width, height, color='gray'))

    # Mark start and goal
    ax.scatter(*start, color="green", s=100, label="Start", edgecolors="black")
    ax.scatter(*goal, color="red", s=100, label="Goal", edgecolors="black")

    # Plot shortest path (if found)
    if path:
        px, py = zip(*path)
        ax.plot(px, py, color='red', linewidth=2, marker='o', markersize=5, label='Path')

    # Plot performance metrics
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
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('PRM Roadmap')
    ax.legend(loc='upper right')
    plt.show()

def main():
    
    grid_size = 19
    start = (4.5, 9.5) # Add 0.5 to be in the middle of the square
    goal = (14.5, 9.5)
    # obstacles = [(9, 1, 10, 18)]  # (x_min, y_min, x_max, y_max)
    obstacles = [(2, 6, 8, 7), (2, 12, 8, 13), (7, 7, 8, 12)]
    max_distance = 1.5
    edge_collision_check_size = 0.05
    max_iterations = 300

    nodes = preprocess(grid_size, max_iterations, obstacles, max_distance, edge_collision_check_size)
    nodes = query(start, goal, nodes, obstacles, max_distance, edge_collision_check_size)
    path = dijkstra(start, goal, nodes)
    visualize_prm(nodes, grid_size, obstacles, start, goal, path)

if __name__ == "__main__":
    main()