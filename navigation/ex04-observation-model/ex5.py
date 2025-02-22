import numpy as np

def prob(mu, sigma):

    # Normal distribution
    prob = (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(- (mu ** 2) / (2 * (sigma**2)))

    return prob

def landmark_observation_model(z, x, b, sigma_r):
    r_hat = np.sqrt((b[0] - x[0])**2 + (b[1] - x[1])**2)
    #phi_hat = np.arctan2(b[1] - x[1], b[0] - x[0])
    p = prob(z - r_hat, sigma_r)
    return p

def observation_likelihood(r, b, gridmap, sigma_r, size):
    
    # Compute likelihood for each cell in the grid
    for x, y in np.ndindex(size, size):
        gridmap[x, y] *= landmark_observation_model(r, [x, y], b, sigma_r)

    return gridmap