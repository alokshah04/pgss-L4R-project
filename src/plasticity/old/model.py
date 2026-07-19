import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import SAC
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Video, display

ENV_NAME    = 'HalfCheetah-v5'
EXPERT_PATH = 'experts/HalfCheetah-v5'

class Trainer:
  expert : SAC
  env : Env

  def __init__(self):
    self.expert : SAC
    self.env : Env
    make_env()
    print(self.env)
    

  def make_env():
    expert = SAC.load(EXPERT_PATH)
    print('Expert policy loaded.')

    env = gym.make(ENV_NAME)
    print(f'Observation space: {env.observation_space}')
    print(f'Action space:      {env.action_space}')
    env.close()