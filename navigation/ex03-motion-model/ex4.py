# add your fancy code here
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def wrap_angle(angle):
    """Wraps an angle to the range [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def inverse_motion_model(prev_pose, cur_pose):
     
    x1, y1, theta1 = prev_pose
    x2, y2, theta2 = cur_pose

    rot1 = np.arctan2(y2 - y1, x2 - x1) - theta1
    rot1 = wrap_angle(rot1)

    trans = np.sqrt((x1 - x2)**2 + (y2 - y1)**2)

    rot2 = theta2 - theta1 - rot1
    rot2 = wrap_angle(rot2)

    return rot1, trans, rot2


def prob(mu, sigma):

    # Normal distribution
    prob = (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(- (mu ** 2) / (2 * (sigma**2)))

    return prob


def motion_model(cur_pose, prev_pose, odom, alpha):

    a1, a2, a3, a4 = alpha
    odom_prev_pose, odom_cur_pose = odom

    rot1, trans, rot2 = inverse_motion_model(odom_prev_pose, odom_cur_pose)
    rot1_hat, trans_hat, rot2_hat = inverse_motion_model(prev_pose, cur_pose)

    p1 = prob(rot1 - rot1_hat, a1*rot1_hat**2 + a2*trans_hat**2)
    p2 = prob(trans - trans_hat, a3*trans_hat**2 + a4*rot1_hat**2 + a4*rot2_hat**2)
    p3 = prob(rot2 - rot2_hat, a1*rot2_hat**2 + a2*trans_hat**2)

    # Project onto 2D space, so ignore p3 or treat p3 == 1
    p = p1 * p2
    return p

def plot_motion_model_posterior(grid_size, resolution, alpha, init_pose, odom):

    # Define initial conditions
    x0, y0, theta0 = init_pose  # Initial pose in map frame

    # Define the grid centered at (2.0, 3.0)
    x_range = np.linspace(x0 - (grid_size // 2) * resolution, 
                          x0 + (grid_size // 2) * resolution, 
                          grid_size)
    
    y_range = np.linspace(y0 - (grid_size // 2) * resolution, 
                          y0 + (grid_size // 2) * resolution, 
                          grid_size)

    # Initialize posterior grid
    posterior = np.zeros((grid_size, grid_size))

    # Sample orientations for integration (approximation)
    theta_samples = np.linspace(-np.pi, np.pi, 36)  # 36 angles for numerical integration

    # Compute probability for each position in the grid
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            # Compute probability by summing over different theta values
            prob_sum = 0
            for theta in theta_samples:
                prob_sum += motion_model([x, y, theta], [x0, y0, theta0], odom, alpha)
            
            # Approximate integral by averaging
            posterior[j, i] = prob_sum / len(theta_samples)  # Normalize over sampled angles

    # Normalize for visualization
    posterior /= np.max(posterior)

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(posterior, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], origin='lower', cmap='binary')
    plt.colorbar(label="Posterior Probability")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Motion Model Posterior $p(x_t | u_t, x_{t-1})$")
    plt.show()


def sample(b):
    tot = 0
    for _ in range(12):
        tot += np.random.uniform(-0.5, 0.5)  # Sum 12 uniform samples
    
    return tot * b


def sample_motion_model(prev_pose, odom, alpha):
    
    a1, a2, a3, a4 = alpha
    odom_prev_pose, odom_cur_pose = odom
    x, y, theta = prev_pose

    rot1, trans, rot2 = inverse_motion_model(odom_prev_pose, odom_cur_pose)

    rot1_hat = rot1 - sample(a1*rot1**2 + a2*trans**2)
    trans_hat = trans - sample(a3*trans**2 + a4*rot1**2 + a4*rot2**2)
    rot2_hat = rot2 - sample(a1*rot2**2 + a2*trans**2)

    x_prime = x + trans_hat * np.cos(theta + rot1_hat)
    y_prime = y + trans_hat * np.sin(theta + rot1_hat)
    theta_prime = theta + rot1_hat + rot2_hat
    theta_prime = wrap_angle(theta_prime)

    return x_prime, y_prime, theta_prime

def plot_sample_motion_model(alpha, init_pose, odoms):

    # Number of samples per step
    samples = 1000

    # Store all samples across time
    all_samples = []

    # Initialize with initial pose
    poses = [init_pose]
    mean_poses = [init_pose]  # This will store the mean, noiseless poses

    for i in range(len(odoms)-1):
        odom = [odoms[i], odoms[i+1]]
        
        # Sample the noisy motion model
        new_samples = [sample_motion_model(poses[-1], odom, alpha) for _ in range(samples)]
        
        # Append samples to the list for plotting later
        all_samples.append(new_samples)
        
        # Store the mean pose for future iterations (this is the noiseless path)
        mean_pose = np.mean(new_samples, axis=0)  
        poses.append(mean_pose)
        mean_poses.append(mean_pose)  # Add mean pose to the noiseless path

    # Plot results
    plt.figure(figsize=(8, 8))

    # Plot noisy samples
    for i, sample_set in enumerate(all_samples):
        x_vals = [s[0] for s in sample_set]
        y_vals = [s[1] for s in sample_set]
        plt.scatter(x_vals, y_vals, alpha=0.2, label=f"Step {i+1}")

    # Plot the noiseless path (line connecting mean poses)
    mean_x = [pose[0] for pose in mean_poses]
    mean_y = [pose[1] for pose in mean_poses]
    plt.plot(mean_x, mean_y, color='red', label="Noiseless Path", linewidth=2)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Sampled Odometry-Based Motion Model")
    plt.legend()
    plt.show()