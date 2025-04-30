import numpy as np
from envs.mujoco_envs import make_env
import time


def evaluate_tree_policy(env_id, tree, n_episodes=5, render=False):
    env = make_env(env_id, render=render)
    total_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = np.array([tree.predict(obs.reshape(1, -1))[0]]) 
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
