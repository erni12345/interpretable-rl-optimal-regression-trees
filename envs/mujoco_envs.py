import gymnasium as gym

def make_env(env_id, render=False):
    if render:
        return gym.make(env_id, render_mode="human")
    else:
        return gym.make(env_id)
