#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0]/map_res) + origin[0];
    new_pose[1] = np.round(pose[1]/map_res) + origin[1];
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def logodds2prob(logodds):
    if logodds == float("inf"):
        return 1
    else:
        prob = (10 ** logodds)/(1 + 10 ** logodds)
        return prob

def prob2logodds(prob):
    if prob == 0:
        return -float("inf")
    else:
        logodds = np.log10(prob/(1 - prob))
        return logodds
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    if cell == endpoint:
        return prob_occ  # The laser endpoint is likely occupied
    else:
        return prob_free  # Cells along the beam path are more likely free
    ## I DON'T GET THIS.
    ## It isn't used in the below function...

def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    """
    Implements occupancy grid mapping using known poses.

    Parameters:
    - ranges_raw: Array of sensor measurements (ranges).
    - poses_raw: Array of robot poses.
    - occ_gridmap: Initial occupancy grid (log-odds form).
    - map_res: Resolution of the map.
    - prob_occ: Probability of occupancy (P(occ)).
    - prob_free: Probability of free space (P(free)).
    - prior: Prior probability of occupancy.

    Returns:
    - Updated occupancy grid (log-odds form).
    """
    log_prior = prob2logodds(prior)
    log_occ = prob2logodds(prob_occ)
    log_free = prob2logodds(prob_free)

    for t in range(len(poses_raw)):  # Loop over all time steps
        robot_cell = poses2cells(poses_raw[t], occ_gridmap, map_res)  # Robot position in grid
        endpoints = ranges2cells(ranges_raw[t], poses_raw[t], occ_gridmap, map_res)  # Sensor measurements in grid

        for endpoint in endpoints.T:
            free_cells = bresenham(robot_cell[0], robot_cell[1], endpoint[0], endpoint[1])

            for cell in free_cells[:-1]:  # Mark free space
                occ_gridmap[cell[0], cell[1]] += log_free - log_prior

            occ_gridmap[endpoint[0], endpoint[1]] += log_occ - log_prior  # Mark occupied

    return occ_gridmap