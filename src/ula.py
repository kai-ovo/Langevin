import numpy as np

def ULA(score, init_pos, step_size, num_iter):
    """
    Unadjusted Langevin Algorithm (ULA)

    Inputs:
    - score: score function
    - init_pos: iniailization
    - step_size: Step size 
    - num_iter: num iteratinos

    outputs:
    - samples: final samples [num_iter, dim]
    """
    position = init_pos
    samples = [position]

    for i in range(num_iter):
        gradient = score(position)
        noise = np.random.normal(0, np.sqrt(2 * step_size), size=position.shape)
        position = position + step_size * gradient + noise
        samples.append(position)

    return np.array(samples)
