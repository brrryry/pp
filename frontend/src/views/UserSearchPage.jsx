import React from 'react';
import UserSearch from '../components/UserSearch';

function UserSearchPage() {
  return (
    <div className="user-search-page container">
      <header className="hero-section text-center">
        <div className="hero-badge">⚡ osu! Performance & Latent Embedding Profiler</div>
        <h1 className="hero-title">Discover Your Next Maps</h1>
        <p className="hero-subtitle">
          Enter an osu! User ID or Username below to ingest top plays, analyze your replay dataset,
          and reveal your personalized map recommendations.
        </p>
      </header>

      <main className="search-main-content">
        <UserSearch />

        <div className="features-grid">
          <div className="feature-card glass-card">
            <span className="feature-icon">📊</span>
            <h3>Replay Dataset Analysis</h3>
            <p>Inspect all ingested replays with mastery scores, accuracy, misses, mods, and star ratings.</p>
          </div>

          <div className="feature-card glass-card">
            <span className="feature-icon">🌫️</span>
            <h3>Latent Mapping</h3>
            <p>2D UMAP projection of map embeddings highlighting recommended beatmaps tailored for you.</p>
          </div>

          <div className="feature-card glass-card">
            <span className="feature-icon">⚡</span>
            <h3>Automated Ingestion</h3>
            <p>Seamlessly fetches top 100 plays via osu! API v2 and indexes beatmap embeddings in real time.</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default UserSearchPage;
