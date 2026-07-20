import torch
from torch.utils.data import Dataset

class BCSequentialData(Dataset):
    def __init__(self, obs, acts, next_obs, next_acts):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)
        self.next_obs = torch.tensor(next_obs, dtype=torch.float32)
        self.next_acts = torch.tensor(next_acts, dtype=torch.float32)
        
    def __len__(self): 
        return len(self.obs)
        
    def __getitem__(self, idx): 
        return self.obs[idx], self.acts[idx], self.next_obs[idx], self.next_acts[idx]