import random
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

# --- RUNTIME MODE PARAMETER ---
# Options: "CONTROL", "DAGGER", "LYAPUNOV_DAGGER", "LYAPUNOV_DAGGER_TWO_STEP", "LYAPUNOV_ONLY", "TWO_STEP_ONLY"
TRAINING_MODE = None

# Trajectory Settings
NUM_TRAIN_TRAJECTORIES = 70
NUM_VAL_TRAJECTORIES = 10
TRAJECTORY_LENGTH = 300

# DAgger Settings
DAGGER_ITERATION_COUNT = 20
DAGGER_ROLLOUT_EPISODES = 5
DAGGER_EPOCHS = 25
CRITIC_PRETRAIN_EPOCHS = 100
CRITIC_REFRESH_EPOCHS = 25

# Data Dimension & Constraints
KNN_SUBSAMPLE_SIZE = 5000
ACTION_MIN = -2.0
ACTION_MAX = 2.0

# Batch Sizes
POLICY_BATCH_SIZE = 256
CRITIC_BATCH_SIZE = 256

# Optimization Learning Rates
POLICY_LR = 1e-3
POLICY_WD = 1e-5

CRITIC_LR = 1e-3
CRITIC_WD = 1e-4
CRITIC_GRAD_CLIP = 1.0
CRITIC_NOISE_SCALE = 0.3

LYAPUNOV_LR = 1e-3
LYAPUNOV_WD = 1e-5

# Loss Multipliers & Algorithmic Weights
TWO_STEP_LOSS_WEIGHT = 1.0
LYAPUNOV_LOSS_WEIGHT = 0.005
LYAPUNOV_ALPHA_DECAY = 0.001

# Contrastive-regularization settings for the Lyapunov network
LYAPUNOV_NOISE_SCALE = 0.3
LYAPUNOV_MARGIN = 1.0
LYAPUNOV_CONTRASTIVE_WEIGHT = 1.0

# Composite-score weights used for checkpoint selection
ENERGY_SCORE_WEIGHT = 50.0
KNN_SCORE_WEIGHT = 20.0

# Evaluation Environments
MID_TRAIN_EVAL_EPISODES = 3
FINAL_EVAL_EPISODES = 10

SEED = 73

def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)