import numpy as np
import matplotlib.pyplot as plt

def rho(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)

def goal_potential(point_1, point_2):
    distance = rho(point_1, point_2)
    return 0.5 * distance ** 2

def obstacle_potential(point_1, point_obstacle, obstacle_radius):
    n = 10000
    min_distance = 25
    distance = rho(point_1, point_obstacle)
    if distance <= obstacle_radius:
        potential = n
    elif distance <= min_distance:
        potential = n * 0.5 * ((1/distance) - (1/min_distance)) ** 1/2
    else:
        potential = 0

    return potential

def generate_world(grid_size, num_obstacles, random_seed):
    np.random.seed(random_seed)
    
    # Start somewhere at top left, end somewhere at bottom right
    start = np.array((np.random.randint(0, 10), np.random.randint(0, 10)))
    goal = np.array((np.random.randint(grid_size-10, grid_size), np.random.randint(grid_size-10, grid_size)))
    print(f"Goal: {goal}")

    # Randomly locate obstacles
    obstacles = []
    for _ in range(num_obstacles):
        obstacles.append(np.array((np.random.randint(0, grid_size), np.random.randint(0, grid_size))))

    return start, goal, obstacles

def calculate_overall_potential(grid_size, obstacles, obstacle_radius, goal):
    
    # Compute potential field
    potential = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            coord = np.array((i, j))
            potential[j, i] += goal_potential(coord, goal)
            for obstacle in obstacles:
                potential[j, i] += obstacle_potential(coord, obstacle, obstacle_radius)

    return potential

def calculate_grad(potential):
    
    dU_dy, dU_dx = np.gradient(potential)

    return dU_dx, dU_dy

def bilinear_interpolation(grid, x, y):
    """
    Perform bilinear interpolation for a given float coordinate (x, y) on a 2D integer grid.
    
    :param grid: 2D NumPy array representing the integer grid.
    :param x: Floating point x-coordinate.
    :param y: Floating point y-coordinate.
    :return: Interpolated value at (x, y).
    """
    h, w = grid.shape
        
    if np.floor(x) == x:
        x0 = int(x)
        x1 = min(x0 + 1, w - 1)
    else:
        x0, x1 = int(np.floor(x)), int(np.ceil(x))
    if np.floor(y) == y: 
        y0 = int(y)
        y1 = min(y0 + 1, h - 1)
    else:
        y0, y1 = int(np.floor(y)), int(np.ceil(y))
    
    # Ensure points are within bounds
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    
    # Get values at the four surrounding grid points
    Q11 = grid[y0, x0]  # Top-left
    Q21 = grid[y0, x1]  # Top-right
    Q12 = grid[y1, x0]  # Bottom-left
    Q22 = grid[y1, x1]  # Bottom-right
    
    # Compute interpolation weights
    dx, dy = x - x0, y - y0
    
    # Interpolate in x direction
    R1 = (1 - dx) * Q11 + dx * Q21
    R2 = (1 - dx) * Q12 + dx * Q22
    
    # Interpolate in y direction
    P = (1 - dy) * R1 + dy * R2
    
    return P

def calculate_new_position(position, dU_dx, dU_dy, step_size, grid_size):
    position_x = position[0]
    position_y = position[1]    

    dU_dx_point = bilinear_interpolation(dU_dx, position_x, position_y)
    dU_dy_point = bilinear_interpolation(dU_dy, position_x, position_y)

    new_position = position.copy()
    new_position[0] = np.clip(new_position[0] + step_size * -dU_dx_point, 0, grid_size - 1)
    new_position[1] = np.clip(new_position[1] + step_size * -dU_dy_point, 0, grid_size - 1)

    return new_position

def check_if_stuck(path, position):
    
    last_pose = path[-1]
    if last_pose[0] == position[0] and last_pose[1] == position[1]:
        return True
    
    last_pose = path[-2]
    if last_pose[0] == position[0] and last_pose[1] == position[1]: 
        return True
    
    return False

def potential_field_planner(start, goal, dU_dx, dU_dy, step_size, grid_size):
    
    distance_to_goal = np.linalg.norm(start - goal)
    path = []
    position = np.array(start, dtype=float)
    path.append(position.copy())

    while distance_to_goal > 10:
        
        position = calculate_new_position(position, dU_dx, dU_dy, step_size, grid_size)
        distance_to_goal = np.linalg.norm(position - goal)

        stuck = False
        if len(path) >= 2:
            stuck = check_if_stuck(path, position)
        if stuck == True:
            print(f"Stuck at {position}!")
            break

        path.append(position.copy())

    if stuck == False:
        path.append(goal.copy())

    return path

def main(grid_size, num_obstacles, obstacle_radius, random_seed, step_size):

    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)

    start, goal, obstacles = generate_world(grid_size, num_obstacles, random_seed)

    potential = calculate_overall_potential(grid_size, obstacles, obstacle_radius, goal)

    dU_dx, dU_dy = calculate_grad(potential)

    path = potential_field_planner(start, goal, dU_dx, dU_dy, step_size, grid_size)

    path_x, path_y = np.array(path)[:, 0], np.array(path)[:, 1]
    path_z = []
    for x, y in zip(path_x, path_y):
        z = bilinear_interpolation(potential, x, y)
        path_z.append(z)

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the potential field surface
    ax.plot_surface(X, Y, potential, cmap="viridis", edgecolor="none", alpha=0.8)

    # Plot the path in 3D
    ax.plot3D(path_x, path_y, path_z, 'r', linewidth=2, label="Path")

    # Plot start and goal points
    ax.scatter(start[0], start[1], potential[start[1], start[0]], color="blue", s=100, label="Start")
    ax.scatter(goal[0], goal[1], potential[goal[1], goal[0]], color="red", s=100, label="Goal")

    # Labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Potential Field Value")
    ax.set_title("Potential Field with Path in 3D")
    ax.legend()

    plt.show()

grid_size = 100
num_obstacles = 10
obstacle_radius = 3
random_seed = 300
step_size = 0.005

main(grid_size, num_obstacles, obstacle_radius, random_seed, step_size)