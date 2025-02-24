# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def world2map(pose, gridmap, map_res):
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)
    return new_pose.astype(int)


def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def t2v(tr):
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    # rays within range
    max_range = 80
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    points = np.array([
        np.multiply(ranges[idx], np.cos(angles[idx])),
        np.multiply(ranges[idx], np.sin(angles[idx]))
    ])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, np.size(points, 1))), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges, r_angles)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # world to map
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose


def init_uniform(num_particles, img_map, map_res):
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map,
                                                              1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map,
                                                              0) * map_res
    particles[:, 2] = np.random.rand(num_particles) * 2 * np.pi
    particles[:, 3] = 1.0
    return particles


def plot_particles(particles, img_map, map_res):
    plt.matshow(img_map, cmap="gray")
    max_y = np.size(img_map, 0) - 1
    xs = np.copy(particles[:, 0]) / map_res
    ys = max_y - np.copy(particles[:, 1]) / map_res
    plt.plot(xs, ys, '.b')
    plt.xlim(0, np.size(img_map, 1))
    plt.ylim(0, np.size(img_map, 0))
    plt.show()


################### solution ###################
def wrapToPi(theta):
    # TODO: theta between -pi and pi
    return theta
    
    
def sample_normal_distribution(b):
    
    tot = 0
    for i in range(12):
        pass
        # TODO
    
    return 0.5*tot


def forward_motion_model(x_robo, del_rot_1, del_trans, del_rot_2):
   
    # forward motion model 
    # TODO: return x, y, theta in homogeneous coordinates!
    return 0

def wrap_angle(angle):
    """Wraps an angle to the range [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def sample(b):
    tot = 0
    for _ in range(12):
        tot += np.random.uniform(-0.5, 0.5)  # Sum 12 uniform samples
    
    return tot * b

def inverse_motion_model(prev_pose, cur_pose):
     
    x1, y1, theta1 = prev_pose
    x2, y2, theta2 = cur_pose

    rot1 = np.arctan2(y2 - y1, x2 - x1) - theta1
    rot1 = wrap_angle(rot1)

    trans = np.sqrt((x1 - x2)**2 + (y2 - y1)**2)

    rot2 = theta2 - theta1 - rot1
    rot2 = wrap_angle(rot2)

    return rot1, trans, rot2

def sample_motion_model_odometry(x_robo_prev, u, noise_parameters):
    
    a1, a2, a3, a4 = noise_parameters
    odom_prev_pose, odom_cur_pose = u
    x, y, theta = x_robo_prev

    rot1, trans, rot2 = inverse_motion_model(odom_prev_pose, odom_cur_pose)

    rot1_hat = rot1 - sample(a1*rot1**2 + a2*trans**2)
    trans_hat = trans - sample(a3*trans**2 + a4*rot1**2 + a4*rot2**2)
    rot2_hat = rot2 - sample(a1*rot2**2 + a2*trans**2)

    x_prime = x + trans_hat * np.cos(theta + rot1_hat)
    y_prime = y + trans_hat * np.sin(theta + rot1_hat)
    theta_prime = theta + rot1_hat + rot2_hat
    theta_prime = wrap_angle(theta_prime)
    
    return x_prime, y_prime, theta_prime

def compute_weights(x_pose, z_obs, gridmap, likelihood_map, map_res):
    num_particles = x_pose.shape[0]
    weights = np.zeros(num_particles)

    for i in range(num_particles):
        # Convert sensor readings into map coordinates
        m_points = ranges2cells(z_obs[1, :], z_obs[0, :], x_pose[i, :3], gridmap, map_res)

        # Filter out points that fall outside the map bounds
        valid_mask = (m_points[0] >= 0) & (m_points[0] < gridmap.shape[1]) & \
                     (m_points[1] >= 0) & (m_points[1] < gridmap.shape[0])
        valid_points = m_points[:, valid_mask]

        # Extract likelihood values for valid points
        likelihood_values = likelihood_map[valid_points[1], valid_points[0]]

        # Compute weight as product of likelihoods
        weights[i] = np.prod(likelihood_values) if likelihood_values.size > 0 else 1e-10  # Avoid zero weight

    # Normalize weights to sum to 1
    weights += 1e-10  # Avoid division by zero
    weights /= np.sum(weights)

    return weights

def resample(particles, weights):
    num_particles = len(particles)
    resampled_particles = np.zeros_like(particles)
    
    # Normalize weights
    weights /= np.sum(weights)
    
    # Compute cumulative sum
    cumulative_sum = np.cumsum(weights)
    
    # Draw a random starting point
    start = np.random.uniform(0, 1 / num_particles)
    
    # Systematic resampling
    index = 0
    for i in range(num_particles):
        u = start + i / num_particles
        while u > cumulative_sum[index]:
            index += 1
        resampled_particles[i] = particles[index]
    
    return resampled_particles


def mc_localization(odom, z, num_particles, particles, noise, gridmap, likelihood_map, map_res, img_map):

    # TODO

    return particles

def mc_localization(odom, z, num_particles, particles, noise, gridmap, likelihood_map, map_res, img_map):
    for t in range(1, len(odom)):  # Start from 1 since t=0 has no previous pose
        odom_prev_pose = odom[t - 1]
        odom_cur_pose = odom[t]
        u = (odom_prev_pose, odom_cur_pose)  # Construct (prev, cur) tuple

        for i in range(num_particles):
            particles[i, :3] = sample_motion_model_odometry(particles[i, :3], u, noise)

        # Compute weights
        weights = np.zeros(num_particles)
        for i in range(num_particles):
            weights[i] = compute_weights(particles[i, :3], z[t], gridmap, likelihood_map, map_res)
        
        # Normalize weights
        weights += 1e-300  # Avoid division by zero
        weights /= np.sum(weights)

        # Resample particles
        particles = resample(particles, weights)

    return particles