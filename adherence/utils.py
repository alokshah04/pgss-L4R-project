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
    """Collects (obs, act, next_obs, next_act) tuples from expert rollouts."""
    obs_list, act_list, next_obs_list, next_act_list = [], [], [], []

    for i in tqdm(range(num_trajectories), desc=desc):
        obs, _ = env.reset(seed=config.SEED + seed_offset + i)
        done = False
        step = 0

        while not done and step < config.TRAJECTORY_LENGTH:
            action, _ = expert.predict(obs, deterministic=True)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            next_action, _ = expert.predict(next_obs, deterministic=True)

            obs_list.append(obs)
            act_list.append(action)
            next_obs_list.append(next_obs)
            next_act_list.append(next_action)

            obs = next_obs
            step += 1

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        np.array(next_obs_list, dtype=np.float32),
        np.array(next_act_list, dtype=np.float32),
    )


def collect_dagger_data(policy, expert, env, beta, obs_mean, obs_std, num_episodes=5, seed_offset=200000):
    """Collects DAgger interaction rollout data mixed with expert actions."""
    policy.eval()
    obs_list, act_list, next_obs_list, next_act_list = [], [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=config.SEED + seed_offset + ep)
        done = False
        step = 0

        while not done and step < config.TRAJECTORY_LENGTH:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
            with torch.no_grad():
                pi_act = policy(obs_t).cpu().numpy()[0]

            expert_act, _ = expert.predict(obs, deterministic=True)

            if np.random.rand() < beta:
                env_act = expert_act
            else:
                env_act = pi_act

            next_obs, reward, term, trunc, _ = env.step(env_act)
            done = term or trunc

            expert_next_act, _ = expert.predict(next_obs, deterministic=True)

            obs_list.append(obs)
            act_list.append(expert_act)
            next_obs_list.append(next_obs)
            next_act_list.append(expert_next_act)

            obs = next_obs
            step += 1

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        np.array(next_obs_list, dtype=np.float32),
        np.array(next_act_list, dtype=np.float32),
    )


def evaluate_policy_with_manifold_tracking(policy, env, nn_detector, obs_mean, obs_std, episodes=3, seed_offset=0):
    policy.eval()
    rewards, knn_distances = [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=config.SEED + seed_offset + ep)
        ep_reward = 0.0
        done = False
        step = 0

        while not done and step < config.TRAJECTORY_LENGTH:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs_t).cpu().numpy()[0]

            distances, _ = nn_detector.kneighbors([obs_norm])
            knn_distances.append(np.mean(distances))

            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc
            step += 1

        rewards.append(ep_reward)

    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(knn_distances)),
        float(np.std(knn_distances)),
    )