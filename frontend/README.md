# 🎨 Frontend Web Application (`frontend/`)

A modern, responsive Single Page Application (SPA) built with **React**, **Vite**, **HTML5 Canvas**, and **Vanilla CSS**. Right now, I can only confirm that this works well on a **Desktop Viewport** - I am working on a mobile-friendly version!

---

## 🌟 Key Features

- **🌫️ Interactive Fog of War Canvas (`FogOfWarGraph`)**: Custom 2D HTML5 canvas renderer with zoom/pan controls, particle animations, radar overlays, and difficulty gradient nodes.
- **👤 Player Profile Dashboard (`UserProfilePage`)**: Comprehensive user profile view showing active ingestion jobs, replay performance metrics, and top matching plays.
- **🔍 Fast User Search (`UserSearchPage`)**: Real-time user search bar with instant navigation.
- **📊 Performance Tables (`ReplaysTable`)**: Filterable table showing mastery scores, accuracy, misses, mods, and star ratings.

---

## 📁 File Structure

```
frontend/
├── index.html                # Entry HTML document
├── vite.config.js            # Vite configuration with proxy rules
├── src/
│   ├── main.jsx              # React DOM entrypoint
│   ├── App.jsx               # React Router layout & routes
│   ├── index.css             # Glassmorphism design system & typography
│   ├── services/
│   │   └── api.js            # Centralized API fetch helper
│   ├── views/
│   │   ├── UserProfilePage.jsx# Full user profile & Fog of War view
│   │   └── UserSearchPage.jsx # User search landing page
│   └── components/
│       ├── FogOfWarGraph.jsx # 2D Canvas graph renderer & tooltip overlay
│       ├── ReplaysTable.jsx  # Interactive replay table component
│       ├── UserSearch.jsx    # Search input component
│       └── Sidebar.jsx       # Layout sidebar navigation
└── dist/                     # Production build destination
```

---

## 🚀 Development & Build Instructions

### Start Vite Development Server:
```bash
# Install Node dependencies
npm install

# Run dev server on port 3000
npm run dev
```

### Build for Production:
```bash
# Compiles production bundle directly to server/static/
npm run build
```

---

## 🌐 Vite Proxy Configuration

During local development (`npm run dev` at `http://localhost:3000`), Vite proxies API requests to the FastAPI backend at `http://localhost:8000`:
- `/users`, `/user`, `/jobs`, `/upload_replay`, `/delete_replay`, `/api` $\rightarrow$ `http://localhost:8000`
- Includes automatic HTML request proxy bypass so browser refreshes on routes like `/users/username` continue serving Vite's SPA `index.html`.

Make sure the backend is running!
