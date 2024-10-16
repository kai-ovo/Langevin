from scipy.stats import multivariate_normal
import numpy as np

def MALA(score, init_pos, step_size, N):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA)

    inputs:
    - score: score function
    - init_pos: initialization
    - step_size: Step size for the update
    - N: Number of iterations

    outputs:
    -samples: [N, dim]
    - accept_rate: The ratio of accepted proposals
    """
    position = init_pos
    samples = [position]
    accept_count = 0

    for _ in range(N):
        gradient = score(position)
        noise = np.random.normal(0, np.sqrt(2 * step_size), size=position.shape)
        proposal = position + step_size * gradient + noise

        curr_prob = multivariate_normal.logpdf(position, mean=proposal)
        proposal_prob = multivariate_normal.logpdf(proposal, mean=position)

        acceptance_prob = min(1, np.exp(proposal_prob - curr_prob))
        if np.random.uniform(0, 1) < acceptance_prob:
            position = proposal
            accept_count += 1

        samples.append(position)

    accept_rate = accept_count / N
    return np.array(samples), accept_rate
