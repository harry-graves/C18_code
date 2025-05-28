#!/usr/bin/env python3
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


def plot_particles(particles, img_map, map_res,s):
    plt.matshow(255-img_map, cmap="Greys")
    max_y = np.size(img_map, 0)-1
    xs = np.copy(particles[:, 0])/map_res
    ys = max_y - np.copy(particles[:, 1])/map_res
    plt.plot(xs, ys, s)
    plt.xlim(0, np.size(img_map, 1))
    plt.ylim(0, np.size(img_map, 0))
    plt.show()


################### solution ###################
def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2*np.pi
    while theta > np.pi:
        theta = theta - 2*np.pi
    return theta
    
    
def sample_normal_distribution(b):
    
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-b,b)
    
    return 0.5*tot


def forward_motion_model(x_robo, del_rot_1, del_trans, del_rot_2):
   
   # forward motion model 
    x_ = x_robo[0] + del_trans*np.cos(x_robo[2] + del_rot_1)
    y_ = x_robo[1] + del_trans*np.sin(x_robo[2] + del_rot_1)
    theta_ = wrapToPi(x_robo[2] + del_rot_1 + del_rot_2)
    
    return np.array([x_, y_, theta_, 1])


def sample_motion_model_odometry(x_robo_prev, u, noise_parameters):
    
    # from odometry parameters
    del_rot_1, del_trans, del_rot_2 = u
    
    #  the relative motion values with some sampled noise
    del_rot_1_hat = del_rot_1 - sample_normal_distribution(noise_parameters[0]*np.abs(del_rot_1) + noise_parameters[1]*np.abs(del_trans))
    del_trans_hat = del_trans - sample_normal_distribution(noise_parameters[2]*np.abs(del_trans) + noise_parameters[3]*np.abs(del_rot_1) + noise_parameters[3]*np.abs(del_rot_2))
    del_rot_2_hat = del_rot_2 - sample_normal_distribution(noise_parameters[0]*np.abs(del_rot_2) + noise_parameters[1]*np.abs(del_trans))
    
    # implementing the forward motion model
    x_t = forward_motion_model(x_robo_prev, del_rot_1_hat, del_trans_hat, del_rot_2_hat)
    
    return x_t


def compute_weights(x_pose, z_obs, gridmap, likelihood_map, map_res):
    
    z_angles = z_obs[0,:]
    z_ranges = z_obs[1,:]
    
    # observation in the gridmap
    z_in_map = ranges2cells(z_ranges, z_angles, x_pose, gridmap, map_res)

    z_map_y = z_in_map[0,:]
    z_map_x = z_in_map[1,:]
    
    # the id where the beam lies outside the gridmap
    idy = (z_map_y < 0) + (z_map_y > gridmap.shape[1]-1)
    idx = (z_map_x < 0) + (z_map_x > gridmap.shape[0]-1)
    
    # number of instances that happens
    num_z_out_map = np.sum(idy + idx)
    # giving all of them the same lesser prob/ weight of 0.1
    weight_ = np.power(0.1, num_z_out_map)
    
    # calculating weights from likelihood map where beam is in gridmap
    weights = likelihood_map[z_map_x[np.logical_not(idx+idy)], z_map_y[np.logical_not(idx+idy)]]
    
    # final weight from product of all weights
    weight = weight_*np.prod(weights)
    
    return weight


def resample(particles, weights):
    """
    Perform low-variance resampling on a set of particles using their importance weights.

    This method generates a new set of particles by selecting from the original ones
    with replacement, in proportion to their weights. It uses the low-variance resampling 
    technique, which minimizes the variance in the number of times a particle is selected,
    improving the stability and performance of particle filters.

    Parameters:
    ----------
    particles : ndarray of shape (N, D)
        The current set of particles, where N is the number of particles and D is the dimension
        of each particle (typically 4: x, y, theta, weight).

    weights : ndarray of shape (N,)
        Normalized importance weights for each particle (must sum to 1).

    Returns:
    -------
    resampled_particles : ndarray of shape (N, D)
        A new set of particles resampled according to the input weights.
    """
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform(0, 1)) / N
    cumulative_sum = np.cumsum(weights)
    indexes = np.zeros(N, dtype=int)

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    resampled_particles = particles[indexes]
    return resampled_particles

def mc_localization(odom, z, num_particles, particles, noise, gridmap, likelihood_map, map_res, img_map):

    # executing for the odometry sequence
    for i in range(len(odom)):        
        weights = np.array([])
        odom_inst = odom[i]
        z_obs = z[i]

        # executing for all the particles in loop
        for m in range(num_particles):
            # forward motion model
            particles[m,:] = sample_motion_model_odometry(particles[m,:], odom_inst, noise)
            # weight computation
            weight = compute_weights(particles[m,:], z_obs, gridmap, likelihood_map, map_res)
            weights = np.append(weights, weight)
        
        # normalise the weights    
        weights = weights/np.sum(weights)
        
        # low variance resampling
        resampled_particles = resample(particles, weights)

        particles = resampled_particles
        
        plot_particles(particles, img_map, map_res, '.b')

    return particles