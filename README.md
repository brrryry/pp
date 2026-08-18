# 🎯 PP Recommender

This repository stores an organized version of code that I wrote to build an Osu! map recommender. This recommender embeds maps using a Variational Autoencoder (VAE) and uses Alternating Least Squares (ALS) to create a recommendation algorithm.

---

## 🌟 Key Features

- **📊 Axis-Free Map Profiling**: Given the raw map data, our embedding model outputs a compressed 128-dimensional vector.
- **🌫️ Map Visualization**: Projects 128D VAE beatmap embeddings onto a interactive 2D spatial map (using Uniform Manifold Approximation and Projection (UMAP)) to visualize mastered and unplayed skill territories.
- **🤖 ALS Recommendation Engine**: Implicit feedback matrix factorization trained on player mastery scores, providing personalized map recommendations.
- **🔁 Retraining Pipeline with Quality Gates**: Automated model retraining pipeline with validation loss threshold checks and deployment archiving.
- **⚡ Dual Storage Backend**: Supports high-performance **PostgreSQL** and lightweight **SQLite** with **Redis** job queues (`rq`) and multi-layer caching.

---

## 📐 Project Architecture

```
pp/
├── server/           # FastAPI backend, DB pool, RQ background workers, API endpoints
├── model_training/   # PyTorch MapVAE training, ALS matrix factorization, seeding scripts
├── frontend/         # React + Vite frontend (Fog of War canvas, User Profiles, Search)
├── static/           # Production static web assets
└── docker-compose.yml # Containerized orchestration (PostgreSQL, Redis, Web App)
```

---

## 🚀 Quickstart & Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **Redis & PostgreSQL** (or Docker)

### 1. Backend Setup (`server/`)
```bash
# Navigate to server directory
cd server

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (copy template)
cp .env.example .env

# Run FastAPI server
python main.py
```
The backend API will run at `http://127.0.0.1:8000` (API docs at `http://127.0.0.1:8000/docs`).

### 2. Frontend Setup (`frontend/`)
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
The frontend dev server will run at `http://localhost:3000` with hot module reloading (HMR) and backend proxying.

---

## 🐳 Docker Deployment

To launch the full stack (PostgreSQL, Redis, and Web App) with containerized environment variables:

```bash
# Build frontend bundle for production
cd frontend && npm install && npm run build && cd ..

# Start containers
docker-compose up --build
```

---

## ⚙️ Environment Configuration

Both `server/` and `model_training/` use environment variables backed by standard `.env` files. See:
- [`server/.env.example`](file:///c:/Users/thisi/Desktop/pp/server/.env.example)
- [`model_training/.env.example`](file:///c:/Users/thisi/Desktop/pp/model_training/.env.example)

Key variables include:
- `DATABASE_URL`: PostgreSQL connection URI (`postgresql://user:pass@host:5432/dbname`)
- `REDIS_URL`: Redis URI (`redis://localhost:6379/0`)
- `OSU_API_CLIENT_ID` / `OSU_API_CLIENT_SECRET`: Official osu! v2 API credentials

---

## 📚 Component Documentation

Detailed guides for individual modules:
- [Backend Documentation (`server/README.md`)](file:///c:/Users/thisi/Desktop/pp/server/README.md)
- [Machine Learning & Pipeline Documentation (`model_training/README.md`)](file:///c:/Users/thisi/Desktop/pp/model_training/README.md)
- [Frontend Web App Documentation (`frontend/README.md`)](file:///c:/Users/thisi/Desktop/pp/frontend/README.md)

---

## 📄 License

Distributed under the MIT License.
