import argparse
import time
from stable_baselines3 import PPO
from envs.mujoco_envs import make_env

def evaluate(env_name, model_path, n_episodes=3, render=True):
    env = make_env(env_name, True)
    model = PPO.load(model_path)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if render:
                env.render()
                time.sleep(1 / 60)  # 60 FPS

        print(f"Episode {ep + 1} reward: {ep_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=["double_pendulum", "hopper"])
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # Map from shorthand to full environment ID
    env_ids = {
        "double_pendulum": "Pendulum-v1",
        "hopper": "Hopper-v4"
    }

    evaluate(env_ids[args.env], args.model_path)
