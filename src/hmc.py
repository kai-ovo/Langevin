# Ref: Neal "MCMC Using Hamiltonian Dynamics" 2011

import numpy as np
def HMC(score, init_pos, step_size, num_leapfrog_steps, N, mass=1.0):
    """
    Hamiltonian Monte Carlo (HMC)

    inputs:
    - score: score of the density
    - init_pos: initialization
    - step_size: Step size for the leapfrog scheme
    - num_leapfrog_steps: Number of leapfrog steps per it
    - N: Number of iterations
    - mass: Mass of the particles (default 1.0)

    outputs:
    - samples: [N, dim]
    """
    position = init_pos
    samples = [position]

    for i in range(N):
        # Sample momentum
        momentum = np.random.normal(0, np.sqrt(mass), size=position.shape)
        initial_momentum = momentum.copy()

        # Leapfrog 
        gradient = score(position)
        momentum = momentum + 0.5 * step_size * gradient
        for _ in range(num_leapfrog_steps):
            position = position + step_size * momentum / mass
            if _ < num_leapfrog_steps - 1:
                gradient = score(position)
                momentum = momentum + step_size * gradient
        gradient = score(position)
        momentum = momentum + 0.5 * step_size * gradient

        # Metropolis-Hastings correction
        proposed_energy = 0.5 * np.sum(momentum ** 2) + score(position)
        initial_energy = 0.5 * np.sum(initial_momentum ** 2) + score(init_pos)
        acceptance_prob = min(1, np.exp(initial_energy - proposed_energy))
        if np.random.uniform(0, 1) < acceptance_prob:
            samples.append(position)
        else:
            position = init_pos

    return np.array(samples)
