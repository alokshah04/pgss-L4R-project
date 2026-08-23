import argparse
import os
import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn

# Import model architecture and utilities from your main codebase
from train import BCPolicy, NearestNeighbors, load_from_hub, TQC, config, utils

def parse_args():
    parser = argparse.ArgumentParser(description="Roll out and evaluate a saved policy checkpoint.")
    parser.add_argument(
        "--policy-path", 
        type=str, 
        required=True, 
        help="Path to the saved PyTorch policy checkpoint (e.g., bc_best_DAGGER.pt)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=10, 
        help="Number of evaluation episodes to roll out (default: 10)"
    )
    parser.add_argument(
        "--render-video", 
        action="store_true", 
        help="Pass this flag to record and export an MP4 video of the rollout"
    )
    parser.add_argument(
        "--output-video", 
        type=str, 
        default="policy_rollout.mp4", 
        help="Output filename for the MP4 video (default: policy_rollout.mp4)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.policy_path):
        raise FileNotFoundError(f"[ERROR] Could not find policy checkpoint at path: {args.policy_path}")

    config.set_seed()
    device = config.DEVICE
    env_id = "Humanoid-v5"

    env = gym.make(env_id)

    print("==================================================")
    print(f" Loading Policy Checkpoint & Statistics...")
    print("==================================================")

    checkpoint_data = torch.load(args.policy_path, map_location=device)

    # Check if checkpoint is a dictionary containing saved normalization parameters
    if isinstance(checkpoint_data, dict) and "obs_mean" in checkpoint_data:
        print("[INFO] Found pre-saved observation statistics inside checkpoint!")
        state_dict = checkpoint_data["policy_state_dict"]
        obs_mean = checkpoint_data["obs_mean"]
        obs_std = checkpoint_data["obs_std"]
        
        # Load expert base data purely for fitting k-NN Reference
        checkpoint_hub = load_from_hub("farama-minari/Humanoid-v5-TQC-medium", "humanoid-v5-TQC-medium.zip")
        expert = TQC.load(checkpoint_hub)
        o_raw, a_raw, _, _ = utils.collect_trajectories(
            env, expert, num_trajectories=config.NUM_TRAIN_TRAJECTORIES, desc="Expert Stats Reference"
        )
    else:
        print("[INFO] Fallback: Calculating observation statistics from base expert dataset...")
        state_dict = checkpoint_data
        checkpoint_hub = load_from_hub("farama-minari/Humanoid-v5-TQC-medium", "humanoid-v5-TQC-medium.zip")
        expert = TQC.load(checkpoint_hub)

        o_raw, a_raw, _, _ = utils.collect_trajectories(
            env, expert, num_trajectories=config.NUM_TRAIN_TRAJECTORIES, desc="Expert Stats Reference"
        )

        obs_mean = o_raw.mean(axis=0)
        obs_std = np.maximum(o_raw.std(axis=0), 1e-2)

    # Enforce clipped observations on manifold reference to avoid k-NN scale corruption
    train_obs_norm = np.clip((o_raw - obs_mean) / obs_std, -10.0, 10.0)

    print("\n[INFO] Fitting k-NN Manifold Reference...")
    nn_detector = NearestNeighbors(n_neighbors=5, algorithm='brute', n_jobs=-1)
    nn_detector.fit(train_obs_norm)

    class SimpleBCPolicy(nn.Module):
        def __init__(self, in_dim, out_dim, h1=256, h2=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, act_dim),
            )

        def forward(self, x):
            return self.net(x)

    obs_dim = o_raw.shape[1]
    act_dim = a_raw.shape[1]

    # 1. Inspect state dict keys from checkpoint
    if isinstance(state_dict, dict) and "policy_state_dict" in state_dict:
        raw_weights = state_dict["policy_state_dict"]
    else:
        raw_weights = state_dict

    # 2. Check if checkpoint comes from legacy 'net' architecture or current 'network'
    is_legacy_net = any(k.startswith("net.") for k in raw_weights.keys())

    if is_legacy_net:
        print("[INFO] Detected legacy/gridsearch architecture ('net'). Instantiating SimpleBCPolicy...")
        # Infer layer sizes from weight tensors
        h1_dim = raw_weights["net.0.weight"].shape[0]
        h2_dim = raw_weights["net.2.weight"].shape[0]
        policy = SimpleBCPolicy(obs_dim, act_dim, h1=h1_dim, h2=h2_dim).to(device)
        policy.load_state_dict(raw_weights)
    else:
        print("[INFO] Detected standard architecture ('network'). Instantiating BCPolicy...")
        policy = BCPolicy(obs_dim, act_dim).to(device)
        policy.load_state_dict(raw_weights)

    policy.eval()

    print(f"\n[INFO] Running {args.episodes} rollout episodes...")
    reward_mean, reward_std, knn_mean, knn_std = utils.evaluate_policy_with_manifold_tracking(
        policy, env, nn_detector, obs_mean, obs_std, episodes=args.episodes
    )

    print("\n================ ROLLOUT RESULTS ================")
    print(f" Evaluated Episodes: {args.episodes}")
    print(f" Mean Cumulative Reward: {reward_mean:.2f} ± {reward_std:.2f}")
    print(f" Mean k-NN Distance:     {knn_mean:.3f} ± {knn_std:.3f}")
    print("==================================================")

    if args.render_video:
        print(f"\n[INFO] Rendering video to {args.output_video}...")
        utils.record_bc_policy(policy, env_id, args.output_video, obs_mean, obs_std)
        print(f"[INFO] Video saved successfully.")

    env.close()

if __name__ == "__main__":
    main()