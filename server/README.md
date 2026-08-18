# 🖥️ Server Module (`server/`)

The backend engine for **osu! PP & Skill Profiler**, built with **FastAPI**, **PyTorch**, **Implicit ALS**, **PostgreSQL**, and **Redis**.

---

## 🌟 Key Responsibilities

1. **API Endpoints**: Fast REST API providing user search, replay analytics, skill diagnostics, map recommendations, and status polling.
2. **Replay Ingestion Pipeline (`ReplayIngestor`)**: Processes `.osr` binary files using `osrparse`, calculates accuracy, combo, and mastery scores, and fetches player top/recent plays asynchronously.
3. **Beatmap Ingestion Pipeline (`BeatmapIngestor`)**: Downloads missing `.osu` beatmap files from mirrors (Nerinyan, OsuDirect, Sayobot), parses hit object timing sequences, and extracts VAE embeddings.
4. **Database & Cache Management (`DatabaseManager`)**: Thread-pooled PostgreSQL / SQLite connection pool with Redis caching (`cache:endpoint:*`, `cache:map:*`) and global rate limiting.
5. **Recommendation Engine (`RecommendationEngine`)**: Cosine similarity matching over ALS item vectors and user latent vectors with star rating tolerance filters (`comfort_sr ± 2.0★`).
6. **2D Map Projection (`MapProjector`)**: Dimensionality reduction (UMAP with PCA fallback) to project 128D VAE embeddings into 2D Fog of War coordinates.

---

## 📁 File Structure

```
server/
├── main.py                          # FastAPI application entrypoint & routing
├── config.py                        # Centralized configuration & environment loader
├── logger_setup.py                  # Dual console & file logger initializer
├── core/
│   ├── DatabaseManager.py           # Database connection pooling & schema operations
│   ├── ReplayIngestor.py            # RQ worker for replay ingestion & osu! API sync
│   ├── BeatmapIngestor.py           # Beatmap downloader, parser, & VAE embedder
│   ├── RecommendationEngine.py      # ALS recommendation matching & filtering
│   ├── Projector.py                 # UMAP 2D coordinate transformer
│   ├── Embedder.py                  # PyTorch MapVAE model architecture
│   ├── ml_parser.py                 # .osu hit object sequence extractor
│   └── parser.py                    # rosu-pp difficulty & performance calculator
├── tests/                           # Unit test suite
├── db_push_to_prod.py               # SQLite to PostgreSQL migration script
└── flush_cache_prod.py              # Redis cache & RQ queue purge utility
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/users?osu_id=<id>` | Resolves user profile and checks active ingestion jobs. |
| `GET` | `/user/replays?osu_id=<id>` | Retrieves user replay dataset; auto-triggers ingestion if replays < 25. |
| `GET` | `/user/recommended_maps?osu_id=<id>&k=10` | Returns top-$k$ personalized map recommendations with 2D coordinates. |
| `POST` | `/user/recalibrate?osu_id=<id>` | Forces a fresh sync of top 100 best plays + 50 recent plays via osu! API v2. |
| `POST` | `/upload_replay` | Uploads an `.osr` file for immediate analysis. |
| `GET` | `/jobs?job_id=<id>` | Polls background ingestion job status. |

---

## ⚙️ Configuration (`.env`)

Configuration is managed globally in [`config.py`](file:///c:/Users/thisi/Desktop/pp/server/config.py) and populated via `.env`:

```env
ENVIRONMENT=development
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DATABASE_URL=postgresql://ppuser:pppassword@localhost:5433/ppdb
REDIS_URL=redis://localhost:6380/0
OSU_API_CLIENT_ID=your_client_id
OSU_API_CLIENT_SECRET=your_client_secret
```

---

## 🧪 Running Tests

To run the pipeline unit tests:
```bash
python -m unittest discover -s tests -p "test_retrain_pipeline.py"
```
