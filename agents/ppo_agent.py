from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import yaml
import os

def load_config(env_name, config_path="config/ppo_config.yaml"):
    with open(config_path, "r") as f:
        all_config = yaml.safe_load(f)
    return all_config[env_name]

def train_ppo(env, env_name, config):
    config["learning_rate"] = float(config["learning_rate"])
    model = PPO(
        config["policy"],
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        verbose=1,
        tensorboard_log=f"logs/{env_name}"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"checkpoints/{env_name}",
        name_prefix="ppo_model"
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=checkpoint_callback
    )
    model.save(f"checkpoints/{env_name}/final_model")
