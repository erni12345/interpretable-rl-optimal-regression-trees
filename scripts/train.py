import argparse
from agents.ppo_agent import load_config, train_ppo
from envs.mujoco_envs import make_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["double_pendulum", "hopper"], required=True)
    args = parser.parse_args()

    config = load_config(args.env)
    env = make_env(config["env_id"])
    train_ppo(env, args.env, config)

if __name__ == "__main__":
    main()
