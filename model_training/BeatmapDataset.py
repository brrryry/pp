"""
BeatmapDataset object for training file.
"""

import os
import sys
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from ml_parser import osu_to_ml_sequence
import torch

class BeatmapDataset(Dataset):
    def __init__(self, beatmaps_folder=None, max_length=2000):
        self.beatmaps_folder = beatmaps_folder or config.MAPS_DIR
        self.max_length = max_length
        self.beatmaps = self._collect_beatmaps()
        print(f"Found {len(self.beatmaps)} beatmaps in {self.beatmaps_folder} (sorted)")
        self.beatmaps.sort() 
    
    def _collect_beatmaps(self):
        beatmaps = []
        for root, dirs, files in os.walk(self.beatmaps_folder):
            for file in files:
                if file.endswith(".osu"):
                    beatmaps.append(os.path.join(root, file))
        return beatmaps

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        beatmap = self.beatmaps[idx]
        output, real_length = osu_to_ml_sequence(beatmap, self.max_length) # return the embedded data.
        
        return torch.tensor(output, dtype=torch.float32), torch.tensor(real_length, dtype=torch.float32)
    