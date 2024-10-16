
# ref: https://proceedings.mlr.press/v32/cheni14.pdf

import numpy as np

def sghmc(log_prob_grad, 
          init_theta, 
          lr=1e-2, 
          alpha=0.1, 
          beta=0.01, 
          n_samples=1000, 
          m=1.0, 
          V=None):
    """
    Stochastic Gradient Hamiltonian Monte Carlo 
    
    inputs:
    - log_prob_grad: score function
    - init_theta: initialization
    - lr: learning rate
    - alpha:  friction 
    - beta:  estimated noise in the stochastic gradient
    - n_samples: number of samples to draw.
    - m: momentum scaling factor
    - V: noise cov

    outputs:
    - samples: (n_samples, dim), final samples
    """
    # Initialize variables
    dim = init_theta.shape[0]
    theta = np.copy(init_theta) 
    momentum = np.zeros_like(theta)  
    samples = np.zeros((n_samples, dim))  

    if V is None:
        V = np.eye(dim)

    C = np.sqrt(2 * (alpha - beta) * lr)

    for i in range(n_samples):
        grad_log_prob = log_prob_grad(theta)
        noise = np.random.multivariate_normal(mean=np.zeros(dim), cov=V)
        momentum = (1 - alpha * lr) * momentum + lr * grad_log_prob + C * noise
        theta += momentum / m
        samples[i] = theta

    return samples
