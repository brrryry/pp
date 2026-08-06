# Osu! PP & Skill Profiler

A self-hosted performance profiler for osu! that analyzes replays and generates
multi-axis skill profiles (Snap Aim, Flow Aim, Speed, Streaming, Stamina, etc.).

## Architecture

```
┌──────────────────────┐          ┌──────────────────────────────────┐
│   Gaming PC          │          │   Server  (Docker on NAS)        │
│                      │  upload  │                                  │
│  companion.py        ├─────────►│  FastAPI   ←──►  SQLite + Redis  │
│  (watches replays)   │  /api/   │  (analysis)     (cache & data)   │
│                      │          │                                  │
│                      │  browse  │  Vite/React                      │
│  Web browser         ◄─────────┤  (frontend)                      │
└──────────────────────┘  :8000   └──────────────────────────────────┘
```

| Component         | Location     | Runs on       |
|-------------------|-------------|---------------|
| Desktop companion | `client/`   | Your gaming PC |
| FastAPI server    | `server/`   | NAS (Docker)   |
| React frontend    | `frontend/` | Built → `static/`, served by FastAPI |
| Feature extraction| `src/`      | Imported by server |

---

## Development (your PC)

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm

### Backend

```bash
# Create venv and install dependencies
python -m venv .venv311
.venv311\Scripts\activate        # Windows
pip install -r requirements.txt

# Start the FastAPI server
cd server
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # Starts Vite dev server on http://localhost:3000
                     # API requests are proxied to http://localhost:8000
```

Open `http://localhost:3000` in your browser during development.
The Vite dev server hot-reloads on file changes.

---

## Production Deployment (NAS)

### What you need on the NAS

- Docker and Docker Compose (most UGreen NAS models support this via the app store)
- SSH or file manager access to copy files

### Step-by-step

#### 1. Copy the project to your NAS

```bash
# Via Git (if your NAS has git)
git clone <your-repo-url> /path/to/pp

# Or copy via SCP / SMB from your PC
scp -r C:\Users\...\pp  user@nas-ip:/path/to/pp
```

#### 2. Build the frontend

The frontend must be compiled before building the Docker image.
You can do this on your PC (recommended) or on the NAS if it has Node.js.

```bash
cd frontend
npm ci                # Clean install from lockfile
npm run build         # Compiles React app → outputs to ../static/
```

After this, `static/` will contain the production-ready `index.html` and
hashed JS/CSS bundles. Copy the built `static/` folder to the NAS if you
built on your PC.

#### 3. Build and start containers

```bash
cd /path/to/pp        # On the NAS
docker-compose up -d --build
```

This starts two containers:

| Container      | Purpose                                    |
|----------------|--------------------------------------------|
| `web`          | FastAPI server on port 8000                |
| `redis-cache`  | In-memory cache for map skills & path hashes |

#### 4. Access the profiler

Open `http://<NAS-IP>:8000` from any device on your network.

### Persistent data

The `server/data/` directory is mounted as a Docker volume
(`./server/data:/app/server/data`), so your SQLite database, replays,
and cached maps survive container rebuilds.

To back up your data, just copy the `server/data/` directory.

### Updating

```bash
cd /path/to/pp

# Pull latest code
git pull

# Rebuild frontend (on PC or NAS)
cd frontend && npm ci && npm run build && cd ..

# Rebuild and restart containers
docker-compose up -d --build
```

### Alternative: Pre-built image transfer

If your NAS has a weak CPU and `docker build` is slow:

```bash
# On your PC
docker build -t osu-profiler .
docker save osu-profiler | gzip > osu-profiler.tar.gz

# Transfer to NAS
scp osu-profiler.tar.gz user@nas-ip:/path/to/

# On the NAS
docker load < osu-profiler.tar.gz
docker-compose up -d
```

---

## Project Structure

```
pp/
├── client/              # Desktop companion (runs on gaming PC)
│   └── companion.py     #   Watches replay folder, uploads to server
├── frontend/            # React + Vite source code
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html       #   Vite HTML entry point
│   └── src/             #   React components and views
├── server/              # FastAPI backend
│   ├── main.py          #   API endpoints and analysis pipeline
│   ├── parser.py        #   .osu file format parser
│   ├── features.py      #   Map feature extraction (11-axis skills)
│   ├── config.py        #   Server configuration
│   └── data/            #   Runtime data (mounted volume in Docker)
│       ├── cache.db     #     SQLite: replay hits, map cache
│       ├── replays/     #     Uploaded .osr files + replays_summary.csv
│       └── maps/        #     Downloaded .osu beatmap files
├── src/                 # Shared source modules
│   └── features.py      #   Feature extraction utilities
├── static/              # Built frontend output (generated by Vite)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```

---

## API Endpoints

| Method | Path                        | Description                          |
|--------|-----------------------------|--------------------------------------|
| POST   | `/api/analyze`              | Upload and analyze a replay + beatmap |
| GET    | `/api/user/{username}`      | Player profile and skill averages    |
| GET    | `/api/map/{beatmap_hash}`   | Beatmap details and leaderboard      |
| GET    | `/api/hits/{replay_basename}` | Raw hit data for diagnostic plots  |
| GET    | `/api/leaderboard`          | Global leaderboard                   |
