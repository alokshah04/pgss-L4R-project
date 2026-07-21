import numpy as np
import torch
import gymnasium as gym
from tqdm.auto import tqdm
import config
import imageio


def record_bc_policy(policy, env_id, filename, obs_mean, obs_std):
    print(f"\nSpawning video environment for {env_id}...")
    video_env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = video_env.reset(seed=config.SEED)
    frames, total_reward = [], 0

    policy.eval()
    for _ in range(1000):
        obs_t = torch.tensor((obs - obs_mean) / obs_std, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = policy(obs_t).cpu().numpy()[0]
        obs, r, term, trunc, _ = video_env.step(action)
        total_reward += r
        frames.append(video_env.render())
        if term or trunc:
            break

    video_env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename} | Total Reward: {total_reward:.2f}")


def collect_trajectories(env, expert, num_trajectories, desc="Expert Rollouts", seed_offset=0):
    """
    Collects (obs, act, next_obs, next_act) tuples from expert rollouts.

    FIX: env.reset() previously had no `seed=`, so every mode invocation (CONTROL, DAGGER,
    LYAPUNOV_ONLY, ...) got a different, uncontrolled initial-state sequence since each mode
    is launched as a separate `python train.py <mode>` process. That adds uncontrolled variance
    to cross-mode comparisons -- a mode "winning" could just mean it got a luckier draw of
    initial states, not that it's actually better. Seeding per-trajectory index (but
    deterministically, off config.SEED) makes data collection reproducible and comparable
    across runs/modes while still varying across trajectories within a run.
    """
    obs_b, act_b, n_obs_b, n_act_b = [], [], [], []
    for traj_idx in tqdm(range(num_trajectories), desc=desc):
        obs, _ = env.reset(seed=config.SEED + seed_offset + traj_idx)
        for _ in range(config.TRAJECTORY_LENGTH):
            action, _ = expert.predict(obs, deterministic=True)
            next_obs, _, term, trunc, _ = env.step(action)
            next_action, _ = expert.predict(next_obs, deterministic=True)

            obs_b.append(obs)
            act_b.append(action)
            n_obs_b.append(next_obs)
            n_act_b.append(next_action)

            obs = next_obs
            if term or trunc:
                break
    return np.array(obs_b), np.array(act_b), np.array(n_obs_b), np.array(n_act_b)


def evaluate_policy_with_manifold_tracking(policy, env, nn_detector, obs_mean, obs_std, episodes=3, seed_offset=0):
    policy.eval()
    all_episode_rewards = []
    all_rollout_distances = []

    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=config.SEED + seed_offset + ep_idx)
        episode_reward = 0
        episode_states = []

        while True:
            # Clip state normalization to prevent extreme numeric spikes
            obs_norm = np.clip((obs - obs_mean) / obs_std, -10.0, 10.0)
            episode_states.append(obs_norm)

            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs_t).cpu().numpy()[0]

            next_obs, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
            obs = next_obs

            if term or trunc:
                break

        all_episode_rewards.append(episode_reward)

        if len(episode_states) > 0:
            episode_states_arr = np.nan_to_num(np.array(episode_states), nan=0.0, posinf=10.0, neginf=-10.0)
            distances, _ = nn_detector.kneighbors(episode_states_arr)
            all_rollout_distances.extend(distances.mean(axis=1))

    return np.mean(all_episode_rewards), np.std(all_episode_rewards), np.mean(all_rollout_distances), np.std(all_rollout_distances)


def collect_dagger_data(policy, expert, env, beta, obs_mean, obs_std, num_episodes=5, seed_offset=0):
    """
    Collects interactive rollout data for the DAgger loop.
    Rolls out using a beta-blend of the expert and the trained policy,
    but labels ALL states with the expert's true actions.

    FIX 1: env.reset() now seeded, same rationale as collect_trajectories.
    FIX 2: episodes are now capped at config.TRAJECTORY_LENGTH steps, matching the initial
    expert-data collection. Previously this loop ran `while True` until the environment's own
    term/trunc, uncapped -- for something like Humanoid-v5 that can be up to the env's internal
    time limit (often much longer than TRAJECTORY_LENGTH). That let dataset growth per DAgger
    iteration vary a lot with policy survival time, independent of anything meaningful about
    policy quality, and made iterations inconsistent with each other in aggregated-data volume.
    """
    obs_b, act_b, n_obs_b, n_act_b = [], [], [], []
    policy.eval()

    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=config.SEED + seed_offset + ep_idx)
        for _ in range(config.TRAJECTORY_LENGTH):
            expert_action, _ = expert.predict(obs, deterministic=True)

            if np.random.rand() < beta:
                action_to_take = expert_action
            else:
                obs_t = torch.tensor((obs - obs_mean) / obs_std, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action_to_take = policy(obs_t).cpu().numpy()[0]

            next_obs, _, term, trunc, _ = env.step(action_to_take)
            expert_next_action, _ = expert.predict(next_obs, deterministic=True)

            obs_b.append(obs)
            act_b.append(expert_action)  # DAgger rule: always log what the expert *would* have done
            n_obs_b.append(next_obs)
            n_act_b.append(expert_next_action)

            obs = next_obs
            if term or trunc:
                break

    return np.array(obs_b), np.array(act_b), np.array(n_obs_b), np.array(n_act_b)