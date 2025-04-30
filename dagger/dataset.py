import numpy as np

def collect_dagger_data(env, expert, tree, steps=1000):
    X, y = [], []
    obs, _ = env.reset()
    step_count = 0

    while step_count < steps:
        # Use tree policy if available, else random
        if tree is not None:
            action = np.array([tree.predict(obs.reshape(1, -1))[0]]) 

        else:
            action = env.action_space.sample()

        # Expert label for current state
        expert_action, _ = expert.predict(obs, deterministic=True)

        X.append(obs)
        y.append(expert_action.item())


        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1

        if done:
            obs, _ = env.reset()

    return np.array(X), np.array(y)
