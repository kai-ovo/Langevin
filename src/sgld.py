import numpy as np
def SGLD(score, init_pos, step_size, N, batch_size):
    """
    Stochastic Gradient Langevin Dynamics (SGLD)

    inputs:
    - score: grad log density
    - init_pos: Initial point 
    - step_size: Step size for the update
    - N: Number of iterations 
    - batch_size: Batch size for gradient estimation

    outputs:
    - samples: [N, dim]
    """
    position = init_pos
    samples = [position]

    for i in range(N):
        gradient = score(position)
        noise = np.random.normal(0, np.sqrt(2 * step_size), size=position.shape)
        position = position + step_size * gradient + noise / np.sqrt(batch_size)
        samples.append(position)

    return np.array(samples)
