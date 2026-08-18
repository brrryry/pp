# 🧠 Machine Learning & Data Pipeline (`model_training/`)

The machine learning module responsible for neural network training, implicit recommendation models, dataset compilation, and automated retraining pipelines.

---

## 🌟 Key Components

1. **Variational Autoencoder (`train_model.py` / `BeatmapDataset.py`)**:
   - Trains a PyTorch sequence autoencoder (`MapVAE`) on hit object features (x, y, time delta, spacing, slider ratios).
   - Encodes complex map patterns into a 128D continuous latent embedding space (`mu` latent representations).

2. **ALS Matrix Factorization (`matrix_factorization.py` / `recommender.py`)**:
   - Computes low-rank implicit feedback matrix factorization (`AlternatingLeastSquares`) using user replay mastery scores:
     $$\text{mastery\_score} = \text{accuracy}^3 \times \max(0, 1 - \frac{\text{misses}}{\text{total}}) \times \frac{\text{max\_combo}}{\text{total}}$$
   - Generates item factor matrices and user factor matrices for cosine similarity matching.

3. **2D UMAP Spatial Projection (`project_maps.py`)**:
   - Trains a `UMAP` (Uniform Manifold Approximation and Projection) model to project 128D map embeddings into 2D $(x, y)$ Fog of War coordinates.

4. **Automated Pipeline Runner (`run_pipeline.py`)**:
   - Multi-stage retraining pipeline orchestrator with quality gates:
     - **Stage 1**: Database ingestion and map status checks.
     - **Stage 2**: Polite, rate-limited beatmap downloading with exponential backoff.
     - **Stage 3**: PyTorch model training.
     - **Stage 4**: Quality Gate evaluation (rejects models if validation loss > max threshold).
     - **Stage 5**: Automated archiving and promotion to production model directories.

5. **Seeding Utilities (`seed/`)**:
   - `seed_maps.py`: Scans and ingests local `.osu` beatmap directory into database tables.
   - `seed_users.py`: Populates top player profiles from osu! API v2.
   - `seed_replays.py`: Processes and seeds replay datasets with calculated mastery metrics.

---

## 📁 File Structure

```
model_training/
├── config.py                 # Hyperparameters & path configurations
├── train_model.py            # PyTorch MapVAE training loop
├── matrix_factorization.py   # Implicit ALS matrix factorization engine
├── recommender.py            # Offline recommender evaluator
├── project_maps.py           # 2D UMAP projection model builder
├── run_pipeline.py           # Automated retraining pipeline CLI
├── fetch_top_scores.py       # Top score fetcher via osu! API v2
├── BeatmapDataset.py         # PyTorch Dataset for sequence padding
├── ml_parser.py              # Sequence parser for .osu hit objects
└── seed/
    ├── seed_maps.py          # Beatmap directory seeder
    ├── seed_users.py         # User profile seeder
    └── seed_replays.py       # Replay dataset seeder
```

---

## 🚀 Running Retraining Pipeline

### Dry Run (Test pipeline execution without heavy GPU training):
```bash
python run_pipeline.py --dry-run --skip-download
```

### Full Retraining Run:
```bash
python run_pipeline.py --env prod --auto-promote
```

### Manual Seeding:
To manually seed, you need your own maps and replays.
```bash
# Seed maps from local directory
python seed/seed_maps.py --maps-dir data/maps

# Seed user top replays
python seed/seed_users.py --count 50
```
