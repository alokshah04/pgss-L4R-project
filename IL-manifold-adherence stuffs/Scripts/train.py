# HOW TO USE DA SCRIPT:
# - - - - - - - - - - - -
# To run the training, first, put the IL-manifold-adherence folder into another folder, as
# some of the packages export videos/graphs into the external directory (the one outside the current one)
# 
# Then, create a new terminal in VS Code or Jupyter Notebook and run this file, train.py, and then follow it with
# either: control, dagger, two_step_only, or some other experimental methods (lyapunov_only, lyapunov_dagger_two_step, lyapunov_dagger)
#
# EX: python IL-manfiold-adherence-SJL/train.py control
# EX 2: python IL-manfiold-adherence-SJL/train.py dagger 
#
# Most of, if not all hyperparams are in the config.py file.
#
# Also, if you change the environment (down in the __name__ == __main__: or something), 
# make sure you delete the energy_critic_expert_cache.pt file and run a control run up until it says the energy critic policy
# has been cached. This is so it's trained on the right manifold every single time.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from stable_baselines3 import SAC
from sb3_contrib import TQC
from huggingface_sb3 import load_from_hub
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Custom project module imports
import config
from models import BCPolicy, EnergyCritic, LyapunovNetwork
from dataset import BCSequentialData
import utils

def plot_research_metrics(stats_history, mode):
    """Generates and saves publication-grade multi-panel performance graphs."""
    epochs = stats_history['global_epoch']
    if not epochs: return

    plt.figure(figsize=(18, 5))
    
    # 1. Training vs Validation Loss Curves
    plt.subplot(1, 3, 1)
    plt.plot(epochs, stats_history['train_loss'], label='Train Loss', color='royalblue', lw=2)
    plt.plot(epochs, stats_history['val_loss'], label='Val Loss', color='orange', lw=2, linestyle='--')
    plt.title('Policy Loss Optimization')
    plt.xlabel('Global Training Epochs')
    plt.ylabel('MSE Loss')
    plt.yscale('log')  # Log scale helps see fine-grained validation convergence
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 2. Manifold Energy Plot with Variance Shadows
    plt.subplot(1, 3, 2)
    mean_energy = np.array(stats_history['energy_mean'])
    std_energy = np.array(stats_history['energy_std'])
    plt.plot(epochs, mean_energy, label='Mean Energy', color='crimson', lw=2)
    plt.fill_between(epochs, mean_energy - std_energy, mean_energy + std_energy, alpha=0.15, color='crimson')
    plt.title('Implicit Manifold Energy vs. Time')
    plt.xlabel('Global Training Epochs')
    plt.ylabel('Energy Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 3. KNN Distance Plot with Variance Shadows (Log-scaled to handle MuJoCo explosions gracefully)
    plt.subplot(1, 3, 3)
    mean_knn = np.array(stats_history['knn_mean'])
    std_knn = np.array(stats_history['knn_std'])
    plt.plot(epochs, mean_knn, label='Mean $k$-NN Distance', color='teal', lw=2)
    plt.fill_between(epochs, mean_knn - std_knn, mean_knn + std_knn, alpha=0.15, color='teal')
    plt.title('State Space $k$-NN Distance (Live Rollouts)')
    plt.xlabel('Global Training Epochs')
    plt.ylabel('Euclidean Distance')
    plt.yscale('log')  # Safely compresses any Epoch 8 style physics explosions
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"manifold_adherence_analysis_{mode}.png", dpi=300)
    plt.close()


import os

def train_energy_critic(obs_raw, acts_raw, obs_mean, obs_std, act_mean, act_std, epochs=None):
    """Loads a cached expert manifold critic if available, otherwise pre-trains and saves it."""
    if epochs is None:
        epochs = config.CRITIC_PRETRAIN_EPOCHS
        
    # Standardized cache filename based on the environment footprint
    critic_cache_path = "energy_critic_expert_cache.pt"
    critic = EnergyCritic(obs_raw.shape[1], acts_raw.shape[1]).to(config.DEVICE)
    
    # Check if we can bypass training entirely
    if os.path.exists(critic_cache_path):
        print(f"\n[INFO] Found cached Energy-Based Manifold Critic ({critic_cache_path}). Loading weights instantly...")
        critic.load_state_dict(torch.load(critic_cache_path, map_location=config.DEVICE))
        critic.eval()
        return critic

    print("\n[INFO] No cache found. Pre-training Energy-Based Manifold Critic from scratch...")
    obs_norm = (obs_raw - obs_mean) / obs_std
    acts_norm = (acts_raw - act_mean) / act_std
    
    tensor_dataset = TensorDataset(
        torch.tensor(obs_norm, dtype=torch.float32), 
        torch.tensor(acts_norm, dtype=torch.float32)
    )
    loader = DataLoader(tensor_dataset, batch_size=config.CRITIC_BATCH_SIZE, shuffle=True)
    
    optimizer = optim.AdamW(critic.parameters(), lr=config.CRITIC_LR, weight_decay=config.CRITIC_WD)
    
    critic.train()
    for epoch in range(epochs):
        total_loss = 0
        for obs, true_acts in loader:
            obs, true_acts = obs.to(config.DEVICE), true_acts.to(config.DEVICE)
            
            # Generate negative samples by perturbing valid actions
            noise = torch.randn_like(true_acts) * config.CRITIC_NOISE_SCALE
            fake_acts = torch.clamp(true_acts + noise, config.ACTION_MIN, config.ACTION_MAX) 
            
            # Bounded Squared-Margin Loss
            loss_true = torch.square(critic(obs, true_acts)).mean()
            loss_fake = torch.square(torch.relu(1.0 - critic(obs, fake_acts))).mean()
            loss = loss_true + loss_fake
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=config.CRITIC_GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Critic Epoch {epoch+1:02d}/{epochs} | Stable Loss: {total_loss/len(loader):.4f}")
    
    # Save to disk so parallel runs or subsequent attempts grab it instantly
    print(f"[INFO] Critic training complete. Saving weights to cache: {critic_cache_path}")
    torch.save(critic.state_dict(), critic_cache_path)
    
    critic.eval()
    return critic

# Change the global tracker to track a composite structural score
best_composite_score = -float("inf")

def train_bc_modular(policy, lyapunov_net, energy_critic, nn_detector, train_data, val_loader, scalar_maps, env, stats_history, epochs=None):
    """Optimized block tracking true losses, silencing loss prints until completion, and tracking standard deviations."""
    if epochs is None:
        epochs = config.BC_TRAIN_EPOCHS
        
    obs_mean, obs_std, act_mean, act_std = scalar_maps
    mode = config.TRAINING_MODE
    
    use_lyapunov = mode in ["LYAPUNOV_DAGGER", "LYAPUNOV_DAGGER_TWO_STEP", "LYAPUNOV_ONLY"]
    use_two_step = mode in ["LYAPUNOV_DAGGER_TWO_STEP", "TWO_STEP_ONLY"]
    
    # --- RESEARCH DIAGNOSTIC BLOCK ---
    print(f"\n-> [ARCHITECTURAL CHECK] Initializing Gradient Descent Engine:")
    print(f"   |-- Sub-Loss: Behavioral Cloning (MSE) -> [ACTIVE]")
    print(f"   |-- Sub-Loss: Multi-Step Lookahead      -> [{'ACTIVE' if use_two_step else 'DISABLED'}]")
    print(f"   |-- Sub-Loss: Lyapunov Safety Metric    -> [{'ACTIVE' if use_lyapunov else 'DISABLED'}]")
    print(f"   +-------------------------------------------------------------")
    
    t_obs_norm = (train_data['obs'] - obs_mean) / obs_std
    t_nobs_norm = (train_data['n_obs'] - obs_mean) / obs_std
    
    dataset = BCSequentialData(t_obs_norm, train_data['acts'], t_nobs_norm, train_data['n_acts'])
    loader = DataLoader(dataset, batch_size=config.POLICY_BATCH_SIZE, shuffle=True, pin_memory=False)
    
    optimizer = optim.AdamW(policy.parameters(), lr=config.POLICY_LR, weight_decay=config.POLICY_WD)
    lyap_optimizer = optim.AdamW(lyapunov_net.parameters(), lr=config.LYAPUNOV_LR, weight_decay=config.LYAPUNOV_WD) if use_lyapunov else None
    mse = nn.MSELoss()
    
    global best_composite_score
    checkpoint_path = f"bc_best_{mode}.pt"
    
    mean_train_loss, mean_val_loss = 0.0, 0.0
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        policy.train()
        if use_lyapunov: lyapunov_net.train()
        
        epoch_train_losses = []
        for o_t, a_t, o_next, a_next in loader:
            o_t, a_t = o_t.to(config.DEVICE), a_t.to(config.DEVICE)
            o_next, a_next = o_next.to(config.DEVICE), a_next.to(config.DEVICE)
            
            loss = mse(policy(o_t), a_t)
            if use_two_step: 
                loss += config.TWO_STEP_LOSS_WEIGHT * mse(policy(o_next), a_next)
            if use_lyapunov:
                v_t, v_next = lyapunov_net(o_t), lyapunov_net(o_next)
                target_decay = config.LYAPUNOV_ALPHA_DECAY * torch.norm(o_t, dim=-1, keepdim=True)
                violation = torch.relu((v_next - v_t) + target_decay)
                loss += config.LYAPUNOV_LOSS_WEIGHT * violation.mean()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            
            if use_lyapunov:
                lyap_loss = lyapunov_net(o_t).mean()
                lyap_optimizer.zero_grad()
                lyap_loss.backward()
                lyap_optimizer.step()

        # --- VALIDATION/LOSS EXTRACTION PHASE ---
        policy.eval()
        epoch_val_losses = []
        epoch_energies = []
        
        with torch.no_grad():
            for o_val, a_val, o_val_next, a_val_next in val_loader:
                o_val, a_val = o_val.to(config.DEVICE), a_val.to(config.DEVICE)
                o_val_next, a_val_next = o_val_next.to(config.DEVICE), a_val_next.to(config.DEVICE)
                
                val_loss = mse(policy(o_val), a_val)
                if use_two_step:
                    val_loss += config.TWO_STEP_LOSS_WEIGHT * mse(policy(o_val_next), a_val_next)
                epoch_val_losses.append(val_loss.item())
                
                pred = policy(o_val)
                pred_norm = (pred - torch.tensor(act_mean, device=config.DEVICE)) / torch.tensor(act_std, device=config.DEVICE)
                energies = energy_critic(o_val, pred_norm).cpu().numpy().flatten()
                epoch_energies.extend(energies)

        # --- LIVE ENVIRONMENT EVALUATION ---
        reward_mean, reward_std, knn_mean, knn_std = utils.evaluate_policy_with_manifold_tracking(
            policy, env, nn_detector, obs_mean, obs_std, episodes=config.MID_TRAIN_EVAL_EPISODES
        )
        
        mean_train_loss = np.mean(epoch_train_losses)
        mean_val_loss = np.mean(epoch_val_losses)
        energy_mean, energy_std = np.mean(epoch_energies), np.std(epoch_energies)
        
        safe_knn = min(knn_mean, 100.0)
        composite_score = reward_mean - (1000.0 * energy_mean) - (100.0 * safe_knn)
        
        stats_history['global_epoch'].append(len(stats_history['global_epoch']) + 1)
        stats_history['train_loss'].append(mean_train_loss)
        stats_history['val_loss'].append(mean_val_loss)
        stats_history['reward_mean'].append(reward_mean)
        stats_history['reward_std'].append(reward_std)
        stats_history['energy_mean'].append(energy_mean)
        stats_history['energy_std'].append(energy_std)
        stats_history['knn_mean'].append(knn_mean)
        stats_history['knn_std'].append(knn_std)
        
        print(f"Epoch {epoch+1:02d} / {epochs} | Reward: {reward_mean:.1f} ± {reward_std:.1f} | "
              f"Energy: {energy_mean:.3f} ± {energy_std:.3f} | "
              f"k-NN Dist: {knn_mean:.3f} ± {knn_std:.3f}")
        
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            print("  --> Checkpoint Saved (New Best Composite Score!)")
            torch.save(policy.state_dict(), checkpoint_path)
            
    print(f"--> Phase Complete | Final Train Loss: {mean_train_loss:.6f} | Final Validation Loss: {mean_val_loss:.6f}\n")
    
    policy.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    return policy, lyapunov_net

if __name__ == "__main__":
    import sys
    
    # 1. Dynamically read the training mode from terminal arguments
    if len(sys.argv) > 1:
        cli_mode = sys.argv[1].upper()
        valid_modes = ["CONTROL", "DAGGER", "LYAPUNOV_DAGGER", "LYAPUNOV_DAGGER_TWO_STEP", "LYAPUNOV_ONLY", "TWO_STEP_ONLY"]
        if cli_mode in valid_modes:
            config.TRAINING_MODE = cli_mode
            print(f"[CLI OVERRIDE] Training Mode successfully set to: {config.TRAINING_MODE}")
        else:
            print(f"[ERROR] '{cli_mode}' is not recognized. Falling back to default.")
            print(f"Valid choices are: {valid_modes}")
    else:
        print(f"[INFO] No terminal argument passed. Using default mode: {config.TRAINING_MODE}")

    # Synchronize iterations and epoch allocations locally based on mode
    is_dagger_mode = config.TRAINING_MODE in ["DAGGER", "LYAPUNOV_DAGGER", "LYAPUNOV_DAGGER_TWO_STEP"]
    DAGGER_ITERATIONS = config.DAGGER_ITERATION_COUNT if is_dagger_mode else 1
    BC_TRAIN_EPOCHS = config.DAGGER_EPOCHS if is_dagger_mode else (config.DAGGER_EPOCHS * config.DAGGER_ITERATION_COUNT)

    checkpoint_filename = f"bc_best_{config.TRAINING_MODE}.pt"
    video_filename = f"behavior_cloning_{config.TRAINING_MODE}.mp4"

    config.set_seed()
    print(f"[START] Pipeline initialized in training mode: {config.TRAINING_MODE}")
    
    ENV_ID = "Humanoid-v5"
    checkpoint = load_from_hub("farama-minari/Humanoid-v5-TQC-medium", "humanoid-v5-TQC-medium.zip")
    expert = TQC.load(checkpoint)
    env = gym.make(ENV_ID)
    
    # Collect Baseline Expert Demonstrations
    o_raw, a_raw, no_raw, na_raw = utils.collect_trajectories(env, expert, config.NUM_TRAIN_TRAJECTORIES, desc="Train Expert Rollouts")
    vo_raw, va_raw, vno_raw, vna_raw = utils.collect_trajectories(env, expert, config.NUM_VAL_TRAJECTORIES, desc="Val Expert Rollouts")
    
    obs_mean, obs_std = o_raw.mean(axis=0), o_raw.std(axis=0) + 1e-8
    act_mean, act_std = a_raw.mean(axis=0), a_raw.std(axis=0) + 1e-8
    scalar_maps = (obs_mean, obs_std, act_mean, act_std)
    
    print("\n[INFO] Spawning k-NN Manifold Reference Engine...")
    train_obs_norm_initial = (o_raw - obs_mean) / obs_std
    nn_detector = NearestNeighbors(n_neighbors=5, algorithm='brute', n_jobs=-1)
    nn_detector.fit(train_obs_norm_initial)
    
    val_obs_norm = (vo_raw - obs_mean) / obs_std
    val_nobs_norm = (vno_raw - obs_mean) / obs_std
    val_loader = DataLoader(BCSequentialData(val_obs_norm, va_raw, val_nobs_norm, vna_raw), batch_size=config.POLICY_BATCH_SIZE, shuffle=False)
    
    # Initialize Networks (Using the caching pre-train feature)
    energy_critic = train_energy_critic(o_raw, a_raw, obs_mean, obs_std, act_mean, act_std, epochs=config.CRITIC_PRETRAIN_EPOCHS)
    lyapunov_net = LyapunovNetwork(o_raw.shape[1]).to(config.DEVICE)
    policy = BCPolicy(o_raw.shape[1], a_raw.shape[1]).to(config.DEVICE)
    
    train_data = {'obs': o_raw, 'acts': a_raw, 'n_obs': no_raw, 'n_acts': na_raw}
    
    stats_history = {
        'global_epoch': [], 
        'train_loss': [], 'val_loss': [],
        'reward_mean': [], 'reward_std': [],
        'energy_mean': [], 'energy_std': [], 
        'knn_mean': [], 'knn_std': []
    }
    
    # SINGLE CLEAN TRAINING LOOP
    for iteration in range(DAGGER_ITERATIONS):
        if DAGGER_ITERATIONS > 1:
            print(f"\n==========================================")
            print(f"  LAUNCHING: DAgger Phase {iteration+1}/{DAGGER_ITERATIONS}")
            print(f"==========================================")
            
        beta = max(0.0, 1.0 - iteration / DAGGER_ITERATIONS)
        
        policy, lyapunov_net = train_bc_modular(
            policy, lyapunov_net, energy_critic, nn_detector, train_data, 
            val_loader, scalar_maps, env, stats_history, epochs=BC_TRAIN_EPOCHS
        )
        
        if is_dagger_mode:
            print(f"[DAgger] Phase Complete. Aggregating environment rollout data...")
            no, na, nno, nna = utils.collect_dagger_data(policy, expert, env, beta, obs_mean, obs_std, num_episodes=config.DAGGER_ROLLOUT_EPISODES)
            
            train_data['obs'] = np.concatenate([train_data['obs'], no], axis=0)
            train_data['acts'] = np.concatenate([train_data['acts'], na], axis=0)
            train_data['n_obs'] = np.concatenate([train_data['n_obs'], nno], axis=0)
            train_data['n_acts'] = np.concatenate([train_data['n_acts'], nna], axis=0)
            
            current_train_norm = (train_data['obs'] - obs_mean) / obs_std
            nn_detector.fit(current_train_norm)
            print(f"[DAgger] Manifold Re-Mapped. New dataset size: {len(train_data['obs'])} steps.")

    print("\n[COMPLETE] Optimization finished. Exporting structural metrics plots...")
    plot_research_metrics(stats_history, config.TRAINING_MODE)
    
    # Final verification run
    policy.load_state_dict(torch.load(f"bc_best_{config.TRAINING_MODE}.pt", map_location=config.DEVICE))
    final_reward, final_std, final_knn_mean, final_knn_std = utils.evaluate_policy_with_manifold_tracking(
        policy, env, nn_detector, obs_mean, obs_std, episodes=config.FINAL_EVAL_EPISODES
    )
    
    final_energies = []
    with torch.no_grad():
        for o_val, _, _, _ in val_loader:
            o_val_dev = o_val.to(config.DEVICE)
            pred = policy(o_val_dev)
            pred_norm = (pred - torch.tensor(act_mean, device=config.DEVICE)) / torch.tensor(act_std, device=config.DEVICE)
            energies = energy_critic(o_val_dev, pred_norm).cpu().numpy().flatten()
            final_energies.extend(energies)
            
    final_energy_mean = np.mean(final_energies)
    final_energy_std = np.std(final_energies)
    
    print("\n==================== FINAL POLICY VERIFICATION ====================")
    print(f"Final Reward:          {final_reward:.2f} ± {final_std:.2f}")
    print(f"Final Manifold Energy: {final_energy_mean:.4f} ± {final_energy_std:.4f}")
    print(f"Final k-NN Distance:   {final_knn_mean:.3f} ± {final_knn_std:.3f}")
    print("===================================================================")
    
    utils.record_bc_policy(policy, ENV_ID, video_filename, obs_mean, obs_std)