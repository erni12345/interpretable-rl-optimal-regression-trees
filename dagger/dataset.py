import numpy as np

def collect_dagger_data(env, expert, trees, steps=1000):
    X, y = [], []
    obs, _ = env.reset()
    step_count = 0
    
    # Get action dimension
    action_dim = env.action_space.shape[0]

    while step_count < steps:
        # Determine action based on tree policy or random sampling
        if action_dim == 1 and trees is not None:
            # Single dimension case with one tree
            obs_reshaped = obs.reshape(1, -1)
            action = np.array([trees.predict(obs_reshaped)[0]])
        elif action_dim > 1 and trees is not None and all(t is not None for t in trees):
            # Multi-dimensional case with one tree per dimension
            action = np.zeros(action_dim)
            obs_reshaped = obs.reshape(1, -1)
            for dim in range(action_dim):
                action[dim] = trees[dim].predict(obs_reshaped)[0]
        else:
            # No trees available or incomplete set of trees, use random action
            action = env.action_space.sample()

        # Expert label for current state
        expert_action, _ = expert.predict(obs, deterministic=True)
        
        X.append(obs)
        # Store full expert action, not just a single value
        y.append(expert_action)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1

        if done:
            obs, _ = env.reset()

    return np.array(X), np.array(y)
