import numpy as np
from envs.mujoco_envs import make_env
import time


def evaluate_tree_policy(env_id, trees, n_episodes=5, render=False):
    env = make_env(env_id, render=render)
    total_rewards = []
    
    # Get action dimension
    action_dim = env.action_space.shape[0]

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            obs_reshaped = obs.reshape(1, -1)
            
            # Determine action based on trees structure
            if action_dim == 1 and not isinstance(trees, list):
                # Single dimension case with one tree
                action = np.array([trees.predict(obs_reshaped)[0]])
            elif action_dim > 1 and isinstance(trees, list):
                # Multi-dimensional case with one tree per dimension
                action = np.zeros(action_dim)
                for dim in range(action_dim):
                    action[dim] = trees[dim].predict(obs_reshaped)[0]
            else:
                # Should not happen if properly set up
                raise ValueError("Trees structure doesn't match action dimension")
                
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if render:
                env.render()
                time.sleep(1/60)

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward = {ep_reward:.2f}")

    env.close()
    avg_reward = np.mean(total_rewards)
    return avg_reward
