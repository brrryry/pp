import sqlite3
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import multiprocessing
from tqdm import tqdm
import sys  

sys.path.append("..")
from pipelines.BeatmapIngestionPipeline import BeatmapIngestionPipeline
from core.parser import parse_osu_file

# ==========================================
# 0. Standalone Multiprocessing Helper
# ==========================================
def _get_active_bpm(obj_time, timing_points):
    active_bpm = 120.0  
    for tp in timing_points:
        if tp['time'] <= obj_time:
            active_bpm = tp.get('current_bpm', active_bpm)
        else:
            break
    return active_bpm

def _process_single_map(args):
    """
    Worker function executed across multiple CPU processes.
    Parses sequential text files and maps them to database target attributes.
    """
    map_hash, star_rating, map_dir = args
    file_path = os.path.join(map_dir, f"{map_hash}.osu")
    
    if not os.path.exists(file_path):
        return None
        
    try:
        # Parse the raw sequential .osu file
        difficulty, timing_points, hit_objects = BeatmapIngestionPipeline(None, None).parse_osu_file(file_path)
        
        # Read file configuration features
        diff_stats = [difficulty['od'], difficulty['cs'], difficulty['ar'], difficulty['hp']]
        seq_matrix = []
        prev_x, prev_y, prev_time = 0, 0, 0
        prev_sv = 1.0

        for i, obj in enumerate(hit_objects):
            if i == 0:
                prev_x, prev_y, prev_time = obj['x'], obj['y'], obj['time']
                prev_sv = obj.get('slider_velocity', prev_sv)
                continue

            x_delta = obj['x'] - prev_x
            y_delta = obj['y'] - prev_y
            t_delta = obj['time'] - prev_time
            obj_type = obj['raw_type']
            
            current_bpm = _get_active_bpm(obj['time'], timing_points)
            sv = obj.get('slider_velocity', prev_sv)

            feature_row = [x_delta, y_delta, t_delta, obj_type, current_bpm, sv] + diff_stats
            seq_matrix.append(feature_row)

            prev_x, prev_y, prev_time = obj['x'], obj['y'], obj['time']
            prev_sv = sv

        if len(seq_matrix) > 0:
            return {
                'features': seq_matrix,
                'target': [star_rating] # Explicitly tracking target from map_stats table
            }
    except Exception as e:
        print(f"Skipping map {map_hash}: {e}")
        
    return None

# ==========================================
# 1. Dataset & Collate Setup
# ==========================================
import os
import torch
from torch.utils.data import Dataset

import os
import torch
from torch.utils.data import Dataset

class OsuBeatmapDataset(Dataset):
    def __init__(self, df, maps_dir="..\\data\\maps", desc="Processing"):
        self.df = df.reset_index(drop=True)
        self.maps_dir = maps_dir
        self.desc = desc
        
        print(f"{desc}: Verifying file paths...")
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            # Match the actual file name convention used on disk
            clean_hash = str(row['map_hash']).strip().lower()
            path = os.path.join(self.maps_dir, str(clean_hash) + ".osu")
            if os.path.exists(path):
                self.valid_indices.append(idx)
                
        print(f"Loaded {len(self.valid_indices)} valid maps out of {len(df)} rows.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        true_idx = self.valid_indices[idx]
        row = self.df.iloc[true_idx]
        
        # Consistent path building using maps_dir and map_hash
        clean_hash = str(row['map_hash']).strip().lower()
        file_path = os.path.join(self.maps_dir, str(clean_hash) + ".osu")
        parsed_data = parse_osu_file(file_path)
        
        hit_objects = parsed_data['hit_objects']
        timing_points = parsed_data['timing_points']
        num_objects = len(hit_objects)
        
        # 1. Initialize an empty tensor with shape (num_objects, 10 features)
        feature_tensor = torch.zeros((num_objects, 10), dtype=torch.float32)
        
        # 2. Fill the tensor using your parsing logic
        for i in range(num_objects):
            # Calculate position and time differences
            if i > 0:
                delta_x = hit_objects[i]['x'] - hit_objects[i-1]['x']
                delta_y = hit_objects[i]['y'] - hit_objects[i-1]['y']
                delta_t = hit_objects[i]['time'] - hit_objects[i-1]['time']
            else:
                delta_x, delta_y, delta_t = 0.0, 0.0, 0.0

            # calculate current bpm and slider velocity using timing points
            for tp in timing_points:
                if tp['time'] <= hit_objects[i]['time']:
                    current_bpm = tp['current_bpm']
                    slider_velocity = tp['slider_velocity']
                    break

            # Assign normalized values directly to the tensor row
            feature_tensor[i, 0] = delta_x / 512.0
            feature_tensor[i, 1] = delta_y / 384.0
            feature_tensor[i, 2] = delta_t / 3000.0
            feature_tensor[i, 3] = float(0 if hit_objects[i]['type'] == 'circle' else 1) / 15.0
            feature_tensor[i, 4] = float(current_bpm) / 300.0
            feature_tensor[i, 5] = float(slider_velocity) / 3.0
            feature_tensor[i, 6] = float(parsed_data['difficulty']['od']) / 10.0
            feature_tensor[i, 7] = float(parsed_data['difficulty']['cs']) / 10.0
            feature_tensor[i, 8] = float(parsed_data['difficulty']['ar']) / 14.0
            feature_tensor[i, 9] = float(parsed_data['difficulty']['hp']) / 10.0

        label = torch.tensor(row['star_rating'], dtype=torch.float32)
        return feature_tensor, label



def pad_collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    targets = torch.stack(targets)
    return padded_seqs, targets, lengths

# ==========================================
# 2. Model Definition
# ==========================================

import os
import torch
from torch.utils.data import Dataset

class OsuStarRatingLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_x)
        return self.fc(hidden[-1])

# ==========================================
# 3. Execution Pipeline with Validation
# ==========================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using target device hardware: {device}")

    # Database Data Import - Joining tables to match map_hash with star_rating
    db_file = '../data/osu_profiler.db'
    conn = sqlite3.connect(db_file)
    
    query = """
        SELECT m.map_hash, ms.star_rating 
        FROM maps m 
        INNER JOIN map_stats ms ON m.map_id = ms.map_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Parallel ingestion loop with clean visualization tags
    train_dataset = OsuBeatmapDataset(train_df, desc="Processing Training Split")
    val_dataset = OsuBeatmapDataset(val_df, desc="Processing Validation Split")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate_fn)

    model = OsuStarRatingLSTM(input_dim=10, hidden_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    print("Beginning Training Routine...")
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for seqs, targets, lengths in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seqs.size(0)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, targets, lengths in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                
                outputs = model(seqs, lengths)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seqs.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_osu_lstm.pt")
            print("Checkpoint saved! Validation loss improved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! Training stopped at epoch {epoch+1}")
                break
