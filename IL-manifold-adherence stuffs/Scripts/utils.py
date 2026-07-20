import numpy as np
import torch
import gymnasium as gym
from tqdm.auto import tqdm
import config
import imageio

# ... keep your other functions (collect_trajectories, evaluate_policy, etc.) ...

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
        if term or trunc: break
        
    video_env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename} | Total Reward: {total_reward:.2f}")

def collect_trajectories(env, expert):
    obs_b, act_b, n_obs_b, n_act_b = [], [], [], []
    for _ in tqdm(range(config.NUM_TRAIN_TRAJECTORIES), desc="Expert Rollouts"):
        obs, _ = env.reset()
        for _ in range(config.TRAJECTORY_LENGTH):
            action, _ = expert.predict(obs, deterministic=True)
            next_obs, _, term, trunc, _ = env.step(action)
            next_action, _ = expert.predict(next_obs, deterministic=True)
            
            obs_b.append(obs)
            act_b.append(action)
            n_obs_b.append(next_obs)
            n_act_b.append(next_action)
            
            obs = next_obs
            if term or trunc: break
    return np.array(obs_b), np.array(act_b), np.array(n_obs_b), np.array(n_act_b)

def collect_trajectories(env, expert, num_trajectories, desc="Expert Rollouts"):
    obs_b, act_b, n_obs_b, n_act_b = [], [], [], []
    for _ in tqdm(range(num_trajectories), desc=desc):
        obs, _ = env.reset()
        for _ in range(config.TRAJECTORY_LENGTH):
            action, _ = expert.predict(obs, deterministic=True)
            next_obs, _, term, trunc, _ = env.step(action)
            next_action, _ = expert.predict(next_obs, deterministic=True)
            
            obs_b.append(obs)
            act_b.append(action)
            n_obs_b.append(next_obs)
            n_act_b.append(next_action)
            
            obs = next_obs
            if term or trunc: break
    return np.array(obs_b), np.array(act_b), np.array(n_obs_b), np.array(n_act_b)

def evaluate_policy_with_manifold_tracking(policy, env, nn_detector, obs_mean, obs_std, episodes=3):
    """
    Evaluates the policy in the live environment and measures geometric 
    manifold adherence per state visited, matching 'Much Ado About Noising'.
    """
    import torch
    import numpy as np
    import config
    
    policy.eval()
    all_episode_rewards = []
    all_rollout_distances = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_states = []
        
        while True:
            # Normalize state for the policy
            obs_norm = (obs - obs_mean) / obs_std
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
        
        # Calculate geometric manifold adherence for EVERY state visited during this specific rollout
        episode_states_arr = np.array(episode_states)
        distances, _ = nn_detector.kneighbors(episode_states_arr)
        
        # Average distance per state for this trajectory
        all_rollout_distances.extend(distances.mean(axis=1))
        
    return np.mean(all_episode_rewards), np.std(all_episode_rewards), np.mean(all_rollout_distances), np.std(all_rollout_distances)


def collect_dagger_data(policy, expert, env, beta, obs_mean, obs_std, num_episodes=5):
    """
    Collects interactive rollout data for the DAgger loop.
    Rolls out using a beta-blend of the expert and the trained policy,
    but labels ALL states with the expert's true actions.
    """
    import torch
    import config
    
    obs_b, act_b, n_obs_b, n_act_b = [], [], [], []
    policy.eval()
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        while True:
            # 1. Always query the expert action to use as the true training label
            expert_action, _ = expert.predict(obs, deterministic=True)
            
            # 2. Decide which action to actually execute using the beta probability schedule
            if np.random.rand() < beta:
                action_to_take = expert_action
            else:
                # Normalize the observation before passing it to your PyTorch policy
                obs_t = torch.tensor((obs - obs_mean) / obs_std, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action_to_take = policy(obs_t).cpu().numpy()[0]
            
            # 3. Take a step in the environment
            next_obs, _, term, trunc, _ = env.step(action_to_take)
            
            # 4. Get the expert's action for the next state (required for Two-Step supervision consistency)
            expert_next_action, _ = expert.predict(next_obs, deterministic=True)
            
            # 5. Append everything to the interactive dataset batch
            obs_b.append(obs)
            act_b.append(expert_action) # DAgger rule: Always log what the expert *would* have done
            n_obs_b.append(next_obs)
            n_act_b.append(expert_next_action)
            
            obs = next_obs
            if term or trunc: 
                break
                
    return np.array(obs_b), np.array(act_b), np.array(n_obs_b), np.array(n_act_b)