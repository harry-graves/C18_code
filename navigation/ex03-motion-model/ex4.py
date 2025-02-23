# add your fancy code here
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def wrap_angle(angle):

    # Wrap the angle to the range [-π, π]
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi

    return wrapped_angle

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