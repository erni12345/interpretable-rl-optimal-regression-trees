import numpy as np
import argparse
from stable_baselines3 import PPO
from pystreed import STreeDPiecewiseLinearRegressor
from envs.mujoco_envs import make_env
from dagger.dataset import collect_dagger_data
from dagger.evaluate_tree import evaluate_tree_policy
import os
import joblib


def dagger_train(env_id, expert_path, n_iters=5, max_depth=3, rollout_steps=1000):
    env = make_env(env_id)
    expert = PPO.load(expert_path)
    
    # Get action dimension
    action_dim = env.action_space.shape[0]
    print(f"Environment action dimension: {action_dim}")
    
    # Init dataset
    X_all = np.empty((0, env.observation_space.shape[0]))
    y_all = np.empty((0, action_dim))

    # For multi-dimensional actions, we'll use one tree per dimension
    trees = [None] * action_dim if action_dim > 1 else None

    for i in range(n_iters):
        print(f"\n=== DAgger Iteration {i+1}/{n_iters} ===")

        # Collect data using current policy (trees or single tree), querying expert
        X_new, y_new = collect_dagger_data(env, expert, trees, steps=rollout_steps)

        # Aggregate dataset
        X_all = np.vstack([X_all, X_new])
        if len(y_all) == 0:
            y_all = y_new
        else:
            y_all = np.vstack([y_all, y_new])

        if action_dim == 1:
            # Single dimension case - just one tree
            tree = STreeDPiecewiseLinearRegressor(
                simple=False,
                max_depth=max_depth,
                cost_complexity=0.01,
                ridge_penalty=0.1
            )
            print(f"Training tree with input shape: {X_all.shape}, output shape: {y_all.flatten().shape}")
            tree.fit(X_all, y_all.flatten())
            
            # Save tree structure
            os.makedirs("trees", exist_ok=True)
            dot_path = f"trees/srt_sl_iter_{i+1}.dot"
            tree.export_dot(dot_path)
            print(f"Saved tree to {dot_path}")
            
            # Save the model
            model_path = f"trees/tree_model_iter_{i+1}.joblib"
            joblib.dump(tree, model_path)
            print(f"Saved model to {model_path}")
            
            # Evaluate tree policy
            avg_rew = evaluate_tree_policy(env_id, tree)
            print(f"Average reward of tree at iteration {i+1}: {avg_rew:.2f}")
        else:
            # Multi-dimensional case - one tree per action dimension
            for dim in range(action_dim):
                print(f"Training tree for action dimension {dim+1}/{action_dim}")
                trees[dim] = STreeDPiecewiseLinearRegressor(
                    simple=False,
                    max_depth=max_depth,
                    cost_complexity=0.01,
                    ridge_penalty=0.1
                )
                # Extract the data for this dimension
                y_dim = y_all[:, dim]
                print(f"Training tree dim {dim} with input shape: {X_all.shape}, output shape: {y_dim.shape}")
                trees[dim].fit(X_all, y_dim)
                
                # Save tree structure
                os.makedirs("trees", exist_ok=True)
                dot_path = f"trees/srt_sl_iter_{i+1}_dim_{dim}.dot"
                trees[dim].export_dot(dot_path)
                print(f"Saved tree for dimension {dim} to {dot_path}")
            
            # Save the ensemble of trees
            # model_path = f"trees/tree_ensemble_iter_{i+1}.joblib"
            # joblib.dump(trees, model_path)
            # print(f"Saved tree ensemble to {model_path}")
            
            # Evaluate tree ensemble policy
            avg_rew = evaluate_tree_policy(env_id, trees)
            print(f"Average reward of tree ensemble at iteration {i+1}: {avg_rew:.2f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True, help="Gym environment ID (e.g., Pendulum-v1)")
    parser.add_argument("--expert_path", type=str, required=True, help="Path to trained PPO expert model")
    parser.add_argument("--n_iters", type=int, default=5, help="Number of DAgger iterations")
    parser.add_argument("--max_depth", type=int, default=3, help="Max depth of regression tree")
    parser.add_argument("--rollout_steps", type=int, default=2000, help="Steps per DAgger rollout")
    parser.add_argument("--simple", action="store_true", help="Use simple linear regression (SRT-SL)")
    args = parser.parse_args()

    dagger_train(
        env_id=args.env_id,
        expert_path=args.expert_path,
        n_iters=args.n_iters,
        max_depth=args.max_depth,
        rollout_steps=args.rollout_steps,
    )
