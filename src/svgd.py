# Based on the paper: "Stein Variational Gradient Descent: 
# A General Purpose Bayesian Inference Algorithm"
# https://arxiv.org/abs/1608.04471

import numpy as np

def rbf_kernel(X, h=-1):
    """
    Compute the RBF kernel and its gradient

    """
    pairwise_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
    if h < 0: 
        h = np.median(pairwise_dists) / np.log(X.shape[0] + 1)
    K = np.exp(-pairwise_dists / (2 * h))
    grad_K = -(X[:, None, :] - X[None, :, :]) * K[..., None] / h
    return K, grad_K

def svgd(particles, grad_log_p, n_iter=100, lr=0.01, kernel_fn=rbf_kernel):
    """
    simple Stein Variational Gradient Descent
    
    inputs:
        - particles: (n_particles, n_dim)
            particles from reference measure
        - grad_log_p: 
            score function
        - n_iter:
            Number of iterations to run SVGD.
        - lr: 
            Learning rate
        - kernel_fn:
            kernel and its grad
    
    outputs:
        particles: (n_particles, n_dim)
    """
    n_particles, _ = particles.shape

    for _ in range(n_iter):
        grad_log_p_vals = np.array([grad_log_p(p) for p in particles])
        K, grad_K = kernel_fn(particles)
        phi = (K @ grad_log_p_vals) / n_particles + np.sum(grad_K, axis=0) / n_particles
        particles += lr * phi

    return particles
