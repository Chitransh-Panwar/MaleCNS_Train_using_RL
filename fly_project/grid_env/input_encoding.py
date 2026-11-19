import numpy as np
import pandas as pd
import torch

class FlyInputEncoder:
    def __init__(self, num_pns, grid_size=16, seed=42):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.num_pns = num_pns
        indices = self.rng.permutation(self.num_pns)
        split_1 = int(0.3 * self.num_pns)
        split_2 = int(0.65 * self.num_pns)
        self.smell_pn_indices = indices[:split_1]
        self.x_pn_indices = indices[split_1:split_2]
        self.y_pn_indices = indices[split_2:]
        
        # 3. Initialize fixed random projection weights for smell PNs
        self.smell_weights = self.rng.uniform(0.5, 2.0, size=(len(self.smell_pn_indices),)).astype(np.float32)
        
        # 4. Initialize random preferred coordinates for x and y tuning curves (bell-shaped Gaussian preference)
        self.x_prefs = self.rng.uniform(0.0, float(grid_size - 1), size=(len(self.x_pn_indices),))
        self.y_prefs = self.rng.uniform(0.0, float(grid_size - 1), size=(len(self.y_pn_indices),))
        self.tuning_width = 2.0 # Spread of the tuning curve
        
    def encode(self, obs, e3_mode=False):
        """
        Encodes observation [smell, x_norm, y_norm] into a full PN activity vector of shape (num_pns,).
        """
        smell, norm_x, norm_y = obs
        x_val = norm_x * (self.grid_size - 1)
        y_val = norm_y * (self.grid_size - 1)
        
        pn_activity = np.zeros(self.num_pns, dtype=np.float32)
        
        # A. Smell PNs encoding
        if e3_mode:
            pn_activity[self.smell_pn_indices] = 0.2 * self.smell_weights
        else:
            pn_activity[self.smell_pn_indices] = smell * self.smell_weights
            
        # B. X-coordinate tuning curves (Gaussian response)
        x_responses = np.exp(-((x_val - self.x_prefs) ** 2) / (2 * (self.tuning_width ** 2)))
        pn_activity[self.x_pn_indices] = x_responses.astype(np.float32)
        
        # C. Y-coordinate tuning curves (Gaussian response)
        y_responses = np.exp(-((y_val - self.y_prefs) ** 2) / (2 * (self.tuning_width ** 2)))
        pn_activity[self.y_pn_indices] = y_responses.astype(np.float32)
        
        return pn_activity