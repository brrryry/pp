import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import random
import optuna

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Assuming these are your custom local modules
from core.ml_parser import osu_to_ml_sequence

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

MAX_LENGTH = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- VAE Reparameterization ---
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# --- Data Augmentation for Contrastive Learning ---
def apply_augmentations(seq, lengths, aug_type='both'):
    """
    seq: (batch, seq_len, input_size)
    lengths: (batch,) original lengths before padding
    Returns augmented version of seq (same shape)
    """
    batch_size, seq_len, feat_dim = seq.shape
    aug_seq = seq.clone()
    # only augment non-padded rows (mask)
    mask = (seq != 0).any(dim=-1)  # (batch, seq_len)
    for i in range(batch_size):
        valid_len = lengths[i].item()
        if valid_len == 0:
            continue
        # Apply augmentations to the valid part
        if aug_type in ('shift', 'both'):
            # Random shift in time (±5% of length)
            shift = random.uniform(-0.05, 0.05) * valid_len
            # Random shift in x, y (±10%)
            shift_x = random.uniform(-0.05, 0.05)
            shift_y = random.uniform(-0.05, 0.05)
            # Apply to the first 3 features (time, x, y)
            # time feature is index 0? Actually input_size=9: we need to know columns.
            # Assuming order: [time, x, y, ...] but we don't know for sure.
            # We'll apply shift to the first three columns (time, x, y).
            aug_seq[i, :valid_len, 0] += shift
            aug_seq[i, :valid_len, 1] += shift_x
            aug_seq[i, :valid_len, 2] += shift_y
        if aug_type in ('scale', 'both'):
            # Random scaling of x, y (±10%)
            scale_x = 1.0 + random.uniform(-0.1, 0.1)
            scale_y = 1.0 + random.uniform(-0.1, 0.1)
            aug_seq[i, :valid_len, 1] *= scale_x
            aug_seq[i, :valid_len, 2] *= scale_y
        if aug_type in ('noise', 'both'):
            # Add small Gaussian noise to all features (except time? maybe all)
            noise_std = 0.01
            noise = torch.randn_like(aug_seq[i, :valid_len, :]) * noise_std
            aug_seq[i, :valid_len, :] += noise
    # Ensure zeros remain zeros (padding)
    aug_seq[~mask] = 0.0
    return aug_seq

# --- Contrastive Loss (InfoNCE) ---
def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2*batch, emb)
    # Similarity matrix
    sim = z @ z.T / temperature
    # Positive mask: (i, i+batch) and (i+batch, i)
    mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    pos_mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=z.device)
    pos_mask[:batch_size, batch_size:] = mask
    pos_mask[batch_size:, :batch_size] = mask
    # Self mask (diagonal)
    self_mask = torch.eye(2*batch_size, dtype=torch.bool, device=z.device)
    # Exponentiate and mask out self
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim * (~self_mask).float()
    # Denominator: sum over all j != i
    denom = exp_sim.sum(dim=1, keepdim=True)
    # Numerator: positive pair similarity (only for the one positive)
    pos_exp = exp_sim[pos_mask].view(2*batch_size, 1)
    # InfoNCE loss
    loss = -torch.log(pos_exp / denom)
    return loss.mean()

# --- Model Definitions (same VAE, but we'll add a projection head) ---
class CNNEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(CNNEmbedder, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=5, stride=5, padding=0),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        self.pool_size = 16
        self.fc_mu = nn.Linear(256 * self.pool_size, output_size)
        self.fc_logvar = nn.Linear(256 * self.pool_size, output_size)

    def compute_length_after_conv(self, length, kernel, stride, padding):
        return torch.floor((length + 2*padding - kernel) / stride) + 1

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)
        features = self.conv_blocks(x)
        final_seq_len = features.size(2)
        device = x.device
        cur_lengths = lengths.clone().float()
        cur_lengths = self.compute_length_after_conv(cur_lengths, 5, 2, 2)
        cur_lengths = self.compute_length_after_conv(cur_lengths, 5, 2, 2)
        cur_lengths = self.compute_length_after_conv(cur_lengths, 5, 2, 2)
        cur_lengths = self.compute_length_after_conv(cur_lengths, 5, 2, 2)
        cur_lengths = self.compute_length_after_conv(cur_lengths, 5, 5, 0)
        cur_lengths = torch.clamp(cur_lengths, min=1).long()
        mask = torch.arange(final_seq_len, device=device)[None, :] < cur_lengths[:, None]
        mask = mask.unsqueeze(1)
        masked_features = features * mask
        pooled = F.adaptive_avg_pool1d(masked_features, self.pool_size)
        scale = final_seq_len / (cur_lengths.float() + 1e-8)
        pooled = pooled * scale[:, None, None]
        pooled = pooled.view(pooled.size(0), -1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

class CNNDecoder(nn.Module):
    def __init__(self, embedding_size, input_size=13):
        super(CNNDecoder, self).__init__()
        self.fc = nn.Linear(embedding_size, 256 * 16)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=125, mode='linear', align_corners=False),
                nn.Conv1d(256, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Upsample(size=250, mode='linear', align_corners=False),
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Upsample(size=500, mode='linear', align_corners=False),
                nn.Conv1d(128, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Upsample(size=1000, mode='linear', align_corners=False),
                nn.Conv1d(64, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Upsample(size=2000, mode='linear', align_corners=False),
                nn.Conv1d(32, input_size, kernel_size=5, padding=2)
            )
        ])

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 16)
        for block in self.dec_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        return x

class MapVAE(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, dropout=0.2, projection_size=64):
        super(MapVAE, self).__init__()
        self.encoder = CNNEmbedder(input_size, hidden_size, embedding_size, dropout=dropout)
        self.decoder = CNNDecoder(embedding_size, input_size)
        # Projection head for contrastive loss
        self.projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, projection_size)
        )

    def forward(self, x, seq_lengths, return_z=False):
        mu, logvar = self.encoder(x, seq_lengths)
        z = reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        if return_z:
            return reconstructed, mu, logvar, z
        else:
            return reconstructed, mu, logvar

    def get_projection(self, z):
        return self.projection(z)