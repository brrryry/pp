"""
Trains a VAE with Contrastive Learning concepts.
Base model is a CNN.
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Integrated for clean terminal progress bars
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# Assuming these are your custom local modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from ml_parser import osu_to_ml_sequence
from BeatmapDataset import BeatmapDataset

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class CNNEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNEmbedder, self).__init__()
        
        # Kept your exact block architecture intact
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Conv1d(256, 256, kernel_size=5, stride=5, padding=0),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # Replace AdaptiveAvgPool1d with custom Masked Pooling logic
        self.pool_size = 16
        self.fc = nn.Linear(256 * self.pool_size, output_size)

    def forward(self, x, lengths):
        # x shape: [batch_size, seq_len, input_size]
        # lengths shape: [batch_size] -> contains actual length of each sequence
        
        # 1. Standard Convolutions
        x = x.permute(0, 2, 1)  # -> [batch_size, input_size, seq_len]
        features = self.conv_blocks(x)  # -> [batch_size, 256, final_seq_len]
        
        final_seq_len = features.size(2)
        device = x.device
        
        # 2. Downsample the dynamic lengths tracking tensor matching the 5 strides:
        # Strides applied: /2, /2, /2, /2, /5 -> Overall reduction factor of 80
        # Formula for standard padding/stride tracking: floor((L + 2P - K)/S) + 1
        cur_lengths = lengths.clone().float()
        
        # Mimic layer transformations mathematically
        cur_lengths = torch.floor((cur_lengths + 2*2 - 5) / 2) + 1 # conv1
        cur_lengths = torch.floor((cur_lengths + 2*2 - 5) / 2) + 1 # conv2
        cur_lengths = torch.floor((cur_lengths + 2*2 - 5) / 2) + 1 # conv3
        cur_lengths = torch.floor((cur_lengths + 2*2 - 5) / 2) + 1 # conv4
        cur_lengths = torch.floor((cur_lengths + 0 - 5) / 5) + 1   # conv5
        
        # Clamp minimum length to 1 to protect short edge cases from collapsing to zero
        cur_lengths = torch.clamp(cur_lengths, min=1).long()
        
        # 3. Create a 1D Mask matching the downsampled timeline
        mask = torch.arange(final_seq_len, device=device)[None, :] < cur_lengths[:, None]
        mask = mask.unsqueeze(1)  # Shape: [batch_size, 1, final_seq_len]
        
        # 4. Enforce Masked Averaging (Instead of standard AdaptiveAvgPool1d)
        # Force all padded index values strictly to 0
        masked_features = features * mask
        
        # Perform dynamic average pooling using actual downsampled sequence lengths
        # Interpolate/Pool the valid items to a fixed width of 16
        pooled = nn.functional.adaptive_avg_pool1d(masked_features, self.pool_size)
        
        # Correct the dilution scale factor caused by standard averaging across zero fields
        # This scales the compressed dimensions cleanly back up to parity
        scale = final_seq_len / cur_lengths.float()
        pooled = pooled * scale[:, None, None]
        
        # 5. Output Projection
        pooled = pooled.view(pooled.size(0), -1)  # -> [batch_size, 256 * 16]
        embedding = self.fc(pooled)
        return embedding


# 2. Define the CNN Decoder (CNNDecoder)
class CNNDecoder(nn.Module):
    def __init__(self, embedding_size, input_size):
        super(CNNDecoder, self).__init__()
        self.fc = nn.Linear(embedding_size, 256 * 16)
        
        # We will upsample and apply Conv1d layers
        self.dec_blocks = nn.ModuleList([
            # 16 -> 32
            nn.Sequential(
                nn.Upsample(size=32, mode='linear', align_corners=False),
                nn.Conv1d(256, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU()
            ),
            # 32 -> 64
            nn.Sequential(
                nn.Upsample(size=64, mode='linear', align_corners=False),
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU()
            ),
            # 64 -> 125
            nn.Sequential(
                nn.Upsample(size=125, mode='linear', align_corners=False),
                nn.Conv1d(128, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU()
            ),
            # 125 -> 250
            nn.Sequential(
                nn.Upsample(size=250, mode='linear', align_corners=False),
                nn.Conv1d(128, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ),
            # 250 -> 500
            nn.Sequential(
                nn.Upsample(size=500, mode='linear', align_corners=False),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ),
            # 500 -> 1000
            nn.Sequential(
                nn.Upsample(size=1000, mode='linear', align_corners=False),
                nn.Conv1d(64, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU()
            ),
            # 1000 -> 2000
            nn.Sequential(
                nn.Upsample(size=2000, mode='linear', align_corners=False),
                nn.Conv1d(32, input_size, kernel_size=5, padding=2)
            )
        ])

    def forward(self, embedding, target_sequence=None):
        # embedding shape: [batch_size, embedding_size]
        x = self.fc(embedding)
        x = x.view(-1, 256, 16)
        
        for block in self.dec_blocks:
            x = block(x)
            
        # x shape: [batch_size, input_size, 2000]
        # Permute back to [batch_size, 2000, input_size]
        x = x.permute(0, 2, 1)
        return x

# Update your Autoencoder Wrapper to use the CNN encoder/decoder components
class MapAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(MapAutoencoder, self).__init__()
        self.encoder = CNNEmbedder(input_size, hidden_size, embedding_size)
        self.decoder = CNNDecoder(embedding_size, input_size)
        
    def forward(self, x, seq_lengths):
        embedding = self.encoder(x, seq_lengths)
        # We pass x into the decoder to match the signature of the LSTM version,
        # though the feed-forward CNN decoder does not need teacher forcing.
        reconstructed = self.decoder(embedding, target_sequence=x)
        return reconstructed, embedding


def compute_masked_loss(reconstructed, target):
    """
    Computes Mean Squared Error while completely ignoring the padded zero-rows 
    generated by the ml_parser.
    """
    # Create binary mask: 1.0 for valid notes, 0.0 if the entire object row is 0
    # Shape: [batch_size, sequence_length]
    mask = (target != 0).any(dim=-1).float()
    
    # Calculate raw element-wise squared differences
    loss_elementwise = (reconstructed - target) ** 2
    
    # Apply mask across all features (expand mask dimensions to [batch_size, sequence_length, 1])
    masked_loss = loss_elementwise * mask.unsqueeze(-1)
    
    # Avoid zero-division errors if a totally blank batch somehow slides in
    total_valid_elements = mask.sum() * target.size(-1)
    if total_valid_elements == 0:
        return torch.tensor(0.0, device=target.device, requires_grad=True)
        
    return masked_loss.sum() / total_valid_elements


def main():
    # --- Hyperparameters ---
    INPUT_SIZE = config.INPUT_SIZE
    HIDDEN_SIZE = config.HIDDEN_SIZE
    EMBEDDING_SIZE = config.EMBEDDING_SIZE
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command line argument for epochs if provided
    if len(sys.argv) > 1:
        try:
            for idx, arg in enumerate(sys.argv):
                if arg == '--epochs' and idx + 1 < len(sys.argv):
                    EPOCHS = int(sys.argv[idx + 1])
                    break
        except ValueError:
            pass

    # --- Dataset and DataLoaders ---
    data_folder = config.MAPS_DIR
    dataset = BeatmapDataset(data_folder, max_length=1000)
    
    if len(dataset) == 0:
        print(f"No beatmaps found in {data_folder}. Skipping training execution.")
        return

    # Split the raw dataset first
    train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    # Wrap the split subsets back into functional DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")

    # Ensure models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # --- Model, Loss Setup, Optimizer ---
    model = MapAutoencoder(INPUT_SIZE, HIDDEN_SIZE, EMBEDDING_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # --- Main Training & Validation Loop ---
    print(f"Starting training on device: {DEVICE}")

    min_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        # Wrap the loader in a tqdm context manager for the training progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")
        
        for batch_data in train_bar:
            batch_data = batch_data.to(DEVICE).float()
            
            # Forward pass
            reconstructed, _ = model(batch_data)
            loss = compute_masked_loss(reconstructed, batch_data)
            
            # Backward pass & Optimization
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clip

            optimizer.step()
            
            train_loss += loss.item() * batch_data.size(0)
            
            # Display the current batch loss directly on the progress bar dynamically
            train_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
        average_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(average_train_loss)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        
        # Wrap the validation loader in a tqdm context manager as well
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch", leave=False)
        
        with torch.no_grad():
            for batch_data in val_bar:
                batch_data = batch_data.to(DEVICE).float()
                reconstructed, _ = model(batch_data)
                loss = compute_masked_loss(reconstructed, batch_data)
                val_loss += loss.item() * batch_data.size(0)
                
        average_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(average_val_loss)
        
        # Print clean epoch summary summaries below the completed progress bars
        print(f"✨ Epoch [{epoch+1}/{EPOCHS}] Final Metrics -> Train Loss: {average_train_loss:.4f} | Val Loss: {average_val_loss:.4f}\n")

        if average_val_loss < min_val_loss:
            min_val_loss = average_val_loss
            save_path = os.path.join(BASE_DIR, "models", f"cnn_osu_embedder_best_{epoch+1}.pth")
            torch.save(model.encoder.state_dict(), save_path)
            print(f"Saved best embedder: {save_path}")

    # Save loss values to a CSV file
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    loss_file = os.path.join(models_dir, "cnn_losses.csv")
    with open(loss_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{t_loss:.6f},{v_loss:.6f}\n")
    print(f"Loss log saved to {loss_file}")

    # Generate and save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Autoencoder Training Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(models_dir, "cnn_loss_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to {plot_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()