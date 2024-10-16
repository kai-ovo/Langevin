import numpy as np

def tempered_langevin(score, 
                      init_theta, 
                      lr=1e-2, 
                      temperature=1.0, 
                      n_samples=1000):
    """
    Tempered Langevin Dynamics (TLD) sampler.
    
    Args:
    - score: score function
    - init_theta: initialization
    - lr: learning rate
    - temperature: temperature parameter that controls the noise scale.
    - n_samples

    Returns:
    - samples: (n_samples, dim)
    """
    dim = init_theta.shape[0]
    theta = np.copy(init_theta) 
    samples = np.zeros((n_samples, dim))

    for i in range(n_samples):
        grad_log_prob = score(theta)
        noise = np.sqrt(2 * lr * temperature) * np.random.normal(size=dim)
        theta += lr * grad_log_prob + noise
        samples[i] = theta

    return samples
