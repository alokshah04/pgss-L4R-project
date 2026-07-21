import random
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

# --- RUNTIME MODE PARAMETER ---
# Options: "CONTROL", "DAGGER", "LYAPUNOV_DAGGER", "LYAPUNOV_DAGGER_TWO_STEP", "LYAPUNOV_ONLY", "TWO_STEP_ONLY"
# NOTE: this is set for real by train.py's CLI-argument handling. Any DAGGER_ITERATIONS /
# BC_TRAIN_EPOCHS computed here at import time is necessarily stale (TRAINING_MODE is still
# None at import), so those two derived values were removed from this file. train.py computes
# them itself, locally, AFTER applying the CLI override -- that is the single source of truth.
TRAINING_MODE = None

# Trajectory Settings
NUM_TRAIN_TRAJECTORIES = 70
NUM_VAL_TRAJECTORIES = 10
TRAJECTORY_LENGTH = 300

# DAgger Settings
DAGGER_ITERATION_COUNT = 20
DAGGER_ROLLOUT_EPISODES = 5
# DAgger rollouts are now capped at the same TRAJECTORY_LENGTH used for the initial expert
# collection (see utils.collect_dagger_data), so aggregated-data volume/composition doesn't
# drift independently of policy quality just because a "while True" loop ran longer for a
# policy that happens to survive longer in the env.

DAGGER_EPOCHS = 25
CRITIC_PRETRAIN_EPOCHS = 25
# Number of extra epochs used to *refresh* (not retrain from scratch) the energy critic on the
# growing DAgger dataset after each aggregation round, so it doesn't stay frozen on only the
# original expert-only distribution while DAgger/Lyapunov policies are scored against it.
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
LYAPUNOV_LOSS_WEIGHT = 0.1
LYAPUNOV_ALPHA_DECAY = 0.01

# Contrastive-regularization settings for the Lyapunov network. Without this, the
# violation loss relu((v_next - v_t) + decay) can be minimized almost as well by a network
# that just learns a near-constant/near-zero V everywhere as by one that learns real
# state-action geometry -- V barely moving satisfies "doesn't increase much" trivially.
# To give V something non-trivial to represent, we additionally train it (mirroring
# EnergyCritic's design) to keep expert-consistent (obs, action) pairs distinctly lower
# than the same obs paired with a perturbed/off-manifold action, by at least LYAPUNOV_MARGIN.
# A constant function scores 0 margin gap and is directly penalized by this term, so it's no
# longer a free minimizer.
LYAPUNOV_NOISE_SCALE = 0.3
LYAPUNOV_MARGIN = 1.0
LYAPUNOV_CONTRASTIVE_WEIGHT = 1.0

# Composite-score weights used for checkpoint selection. These were previously (1000, 100),
# which let the (frozen, potentially out-of-distribution) energy critic term dominate the
# reward term almost completely -- structurally favoring policies that stay close to the
# original expert manifold, i.e. favoring CONTROL-like behavior regardless of actual policy
# quality. Rebalanced to more comparable magnitudes; tune per-environment if reward scale
# changes a lot.
ENERGY_SCORE_WEIGHT = 50.0
KNN_SCORE_WEIGHT = 20.0

# Evaluation Environments
MID_TRAIN_EVAL_EPISODES = 3
FINAL_EVAL_EPISODES = 10

SEED = 213


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)