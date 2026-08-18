import React, { useState, useMemo, useRef } from 'react';

function FogOfWarGraph({ recommendedMaps = [], isLoading = false, username = 'Player' }) {
  const [selectedMap, setSelectedMap] = useState(null);
  const [hoveredMap, setHoveredMap] = useState(null);
  const containerRef = useRef(null);

  // Pure Polar Radar Transformation
  // Radius r = Recommendation Match Quality (#1 top match is closest to origin at 50, 50)
  // Angle theta = Skill Sector (Aim, Speed, Tech, Stamina) or derived feature direction
  const polarNodes = useMemo(() => {
    if (!recommendedMaps || recommendedMaps.length === 0) return [];

    const N = recommendedMaps.length;
    const MIN_CANVAS_RADIUS = 12.0; // Inner #1 match zone (12% SVG radius)
    const MAX_CANVAS_RADIUS = 38.0; // Outer recommendation radar boundary (38% SVG radius)

    const nodes = recommendedMaps.map((m, rankIndex) => {
      // 1. Calculate Radius r directly from Recommendation Rank (Closer = Better Match)
      const rankRatio = N > 1 ? rankIndex / (N - 1) : 0.0;
      const r = MIN_CANVAS_RADIUS + rankRatio * (MAX_CANVAS_RADIUS - MIN_CANVAS_RADIUS);

      // 2. Calculate Angle theta based on map skill features or derived feature direction
      let angle;
      if (m.aim_score !== undefined && m.speed_score !== undefined) {
        angle = Math.atan2((m.aim_score ?? 0.5) - 0.5, (m.speed_score ?? 0.5) - 0.5);
      } else if (m.coord_x !== undefined && m.coord_y !== undefined && (Math.abs(m.coord_x) > 1e-4 || Math.abs(m.coord_y) > 1e-4)) {
        angle = Math.atan2(m.coord_y, m.coord_x);
      } else {
        angle = (rankIndex / N) * 2 * Math.PI - Math.PI / 2;
      }

      // Convert polar (r, angle) to canvas coordinates (cx, cy)
      const cx = 50 + r * Math.cos(angle);
      // Invert Y for SVG coordinates so +Y direction is North/Top
      const cy = 50 - r * Math.sin(angle);

      return {
        map: m,
        rankIndex,
        r,
        angle,
        cx,
        cy
      };
    });

    // 2. Ensure Top 3 nodes occupy distinct angular sectors
    for (let i = 0; i < Math.min(3, nodes.length); i++) {
      for (let j = i + 1; j < Math.min(3, nodes.length); j++) {
        let diff = nodes[j].angle - nodes[i].angle;
        diff = Math.atan2(Math.sin(diff), Math.cos(diff)); // Normalize to [-pi, pi]
        if (Math.abs(diff) < 0.45) { // If closer than ~26 degrees
          const nudge = (0.45 - Math.abs(diff)) / 2 * (diff >= 0 ? 1 : -1);
          nodes[j].angle += nudge;
          nodes[i].angle -= nudge;
          nodes[j].cx = 50 + nodes[j].r * Math.cos(nodes[j].angle);
          nodes[j].cy = 50 - nodes[j].r * Math.sin(nodes[j].angle);
          nodes[i].cx = 50 + nodes[i].r * Math.cos(nodes[i].angle);
          nodes[i].cy = 50 - nodes[i].r * Math.sin(nodes[i].angle);
        }
      }
    }

    // Iterative collision resolution algorithm to guarantee circles do not overlap each other OR the center origin
    const MIN_DIST = 6.4; // Minimum distance for normal nodes
    const TOP_3_MIN_DIST = 9.4; // Increased distance threshold for Top 3 glowing beacon nodes
    const CENTER_MIN_DIST = 9.5; // Minimum distance from center origin (50, 50)
    const ITERATIONS = 50;

    for (let iter = 0; iter < ITERATIONS; iter++) {
      let overlapsFound = false;

      // 1. Keep nodes cleared away from center origin (50, 50)
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const dx = n.cx - 50;
        const dy = n.cy - 50;
        const distToCenter = Math.sqrt(dx * dx + dy * dy);

        if (distToCenter < CENTER_MIN_DIST) {
          overlapsFound = true;
          const angle = distToCenter > 1e-4 ? Math.atan2(dy, dx) : ((i / N) * 2 * Math.PI - Math.PI / 2);
          n.cx = 50 + Math.cos(angle) * CENTER_MIN_DIST;
          n.cy = 50 + Math.sin(angle) * CENTER_MIN_DIST;
        }
      }

      // 2. Resolve node-to-node overlaps (with larger threshold for Top 3 glowing nodes)
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = b.cx - a.cx;
          const dy = b.cy - a.cy;
          const dist = Math.sqrt(dx * dx + dy * dy);

          const requiredDist = (a.rankIndex < 3 || b.rankIndex < 3) ? TOP_3_MIN_DIST : MIN_DIST;

          if (dist < requiredDist) {
            overlapsFound = true;
            const overlap = (requiredDist - dist) / 2;
            const angle = dist > 1e-4 ? Math.atan2(dy, dx) : ((i / N) * 2 * Math.PI);
            const nx = Math.cos(angle);
            const ny = Math.sin(angle);

            a.cx -= nx * overlap;
            a.cy -= ny * overlap;
            b.cx += nx * overlap;
            b.cy += ny * overlap;
          }
        }
      }
      if (!overlapsFound) break;
    }

    return nodes;
  }, [recommendedMaps]);

  const handleNodeClick = (mapItem) => {
    setSelectedMap(mapItem);
  };

  const getSRColor = (sr) => {
    if (!sr) return '#4A5568';
    if (sr < 2.0) return '#48BB78';
    if (sr < 3.5) return '#4299E1';
    if (sr < 5.0) return '#ECC94B';
    if (sr < 6.5) return '#ED8936';
    if (sr < 8.0) return '#F56565';
    return '#9F7AEA';
  };

  function getModsString(modsBitmask) {
    if (!modsBitmask) return '';
    const modMap = {
      0: 'NM', 1: 'NF', 2: 'EZ', 4: 'TD', 8: 'HR', 16: 'DT',
      32: 'RL', 64: 'HT', 128: 'NC', 256: 'FL', 512: 'SO',
      1024: 'AP', 2048: 'PF', 4096: '4K', 8192: '5K', 16384: '6K',
      32768: '7K', 65536: '8K', 131072: 'FI', 262144: 'RD',
      524288: 'LI', 1048576: 'PF', 2097152: 'SG', 4194304: 'TP',
      8388608: 'AP', 16777216: 'CL', 33554432: 'SS', 67108864: 'SM',
    };
    const mods = [];
    for (const [bit, name] of Object.entries(modMap)) {
      if (modsBitmask & Number(bit)) {
        mods.push(name);
      }
    }
    return mods.length ? mods.join(' ') : '';
  }

  return (
    <div className="fog-of-war-container glass-card">
      <div className="fog-header">
        <div className="fog-title">
          <h3>🎯 Polar Skill Recommendation Radar</h3>
          <p className="fog-subtitle">
            Pure Polar Skill Radar for <strong>{username}</strong>. Distance from origin represents <strong>Recommendation Match Quality</strong> (#1 is closest to center). Direction represents <strong>Skill Sector</strong>.
          </p>
        </div>
        <div className="fog-legend">
          <span className="legend-item">
            <span className="dot" style={{ backgroundColor: '#FF66AA' }}></span>
            Skill Center (Bulls-eye)
          </span>
          <span className="legend-item"><span className="dot" style={{ backgroundColor: '#48BB78' }}></span> &lt;3.5★</span>
          <span className="legend-item"><span className="dot" style={{ backgroundColor: '#ECC94B' }}></span> 3.5★-5★</span>
          <span className="legend-item"><span className="dot" style={{ backgroundColor: '#F56565' }}></span> 5★-8★</span>
          <span className="legend-item"><span className="dot" style={{ backgroundColor: '#9F7AEA' }}></span> 8★+</span>
        </div>
      </div>

      <div className="fog-canvas-wrapper" ref={containerRef}>
        {isLoading ? (
          <div className="fog-loading-overlay">
            <span className="spinner">🌀</span>
            <p>Computing polar skill radar graph...</p>
          </div>
        ) : recommendedMaps.length === 0 ? (
          <div className="fog-empty-msg">
            <span>🔍</span>
            <p>No map recommendations generated yet. Ingest replays to reveal your skill radar!</p>
          </div>
        ) : (
          <svg className="fog-svg-map" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
            <defs>
              <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="0.5" />
              </pattern>
              <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="1.2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
              <filter id="goldGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2.0" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>
            <rect width="100" height="100" fill="url(#grid)" />

            {/* Concentric Polar Match Zone Rings */}
            <circle cx="50" cy="50" r="14.0" fill="none" stroke="rgba(255, 102, 170, 0.25)" strokeWidth="0.35" strokeDasharray="1.5,1" />
            <circle cx="50" cy="50" r="26.0" fill="none" stroke="rgba(255, 102, 170, 0.18)" strokeWidth="0.3" strokeDasharray="1.5,1" />
            <circle cx="50" cy="50" r="38.0" fill="none" stroke="rgba(255, 102, 170, 0.12)" strokeWidth="0.3" strokeDasharray="1.5,1" />

            {/* Crosshair Axes centered at (50, 50) */}
            <line x1="6" y1="50" x2="94" y2="50" stroke="rgba(255, 255, 255, 0.08)" strokeWidth="0.25" strokeDasharray="1,1" />
            <line x1="50" y1="6" x2="50" y2="94" stroke="rgba(255, 255, 255, 0.08)" strokeWidth="0.25" strokeDasharray="1,1" />

            {/* Connecting Energy Beams from Skill Center (50, 50) to Top 3 Recommended Nodes */}
            {polarNodes.slice(0, 3).map((n, idx) => {
              const beamColors = ['#00F3FF', '#FFD700', '#FF66AA'];
              const beamWidths = [0.8, 0.6, 0.5];
              return (
                <g key={`beam-${idx}`}>
                  <line
                    x1="50"
                    y1="50"
                    x2={n.cx}
                    y2={n.cy}
                    stroke={beamColors[idx]}
                    strokeWidth={beamWidths[idx]}
                    opacity="0.65"
                    strokeDasharray="2,1.2"
                  >
                    <animate attributeName="stroke-dashoffset" from="12" to="0" dur="1.8s" repeatCount="indefinite" />
                  </line>
                </g>
              );
            })}

            {/* Radial Rays to Other Nodes */}
            {polarNodes.slice(3).map((n, idx) => (
              <line
                key={`ray-${idx}`}
                x1="50"
                y1="50"
                x2={n.cx}
                y2={n.cy}
                stroke="rgba(255, 255, 255, 0.06)"
                strokeWidth="0.2"
              />
            ))}

            {/* Render Recommendation Nodes */}
            {polarNodes.map((n, idx) => {
              const mapItem = n.map;
              const color = getSRColor(mapItem.star_rating);
              const isSelected = selectedMap && selectedMap.map_hash === mapItem.map_hash;
              const isHovered = hoveredMap && hoveredMap.map_hash === mapItem.map_hash;

              const isTop3 = n.rankIndex < 3;
              const topBadgeColors = ['#00F3FF', '#FFD700', '#FF66AA'];

              const r = isTop3 ? (isSelected ? 3.8 : isHovered ? 3.4 : 3.0) : (isSelected ? 3.4 : isHovered ? 3.0 : 2.6);

              return (
                <g key={mapItem.map_hash || idx} className="node-group">
                  {/* Pulsing Beacon Halo Ring for Top 3 & Selected/Hovered Nodes */}
                  {(isTop3 || isSelected || isHovered) && (
                    <circle
                      cx={n.cx}
                      cy={n.cy}
                      r={r * 1.5}
                      fill="none"
                      stroke={isTop3 ? topBadgeColors[n.rankIndex] : color}
                      strokeWidth="0.4"
                      opacity="0.75"
                      className="pulse-ring"
                    />
                  )}

                  {/* Main Node Circle */}
                  <circle
                    cx={n.cx}
                    cy={n.cy}
                    r={r}
                    fill={color}
                    stroke={isTop3 ? topBadgeColors[n.rankIndex] : (isSelected ? '#FFFFFF' : 'rgba(255,255,255,0.5)')}
                    strokeWidth={isTop3 ? 0.9 : (isSelected ? 0.8 : 0.4)}
                    filter={isTop3 ? "url(#goldGlow)" : "url(#glow)"}
                    className={`map-node ${isSelected ? 'selected' : ''}`}
                    onClick={() => handleNodeClick(mapItem)}
                    onMouseEnter={() => setHoveredMap(mapItem)}
                    onMouseLeave={() => setHoveredMap(null)}
                  />

                  {/* Rank Badge Text Outside Below Node Circle */}
                  <text
                    x={n.cx}
                    y={n.cy + r + 2.6}
                    fill={isTop3 ? topBadgeColors[n.rankIndex] : '#FFFFFF'}
                    fontSize={isTop3 ? '2.6' : '2.0'}
                    fontWeight="bold"
                    textAnchor="middle"
                    pointerEvents="none"
                    style={{ textShadow: '0 1px 4px rgba(0,0,0,0.95)' }}
                  >
                    #{n.rankIndex + 1}
                  </text>
                </g>
              );
            })}

            {/* Player Skill Center at Exact Canvas Center (50, 50) */}
            <circle
              cx="50"
              cy="50"
              r="1.4"
              fill="#FF66AA"
              stroke="#FFFFFF"
              strokeWidth="0.5"
              opacity="1"
            />

            <circle
              cx="50"
              cy="50"
              r="12"
              fill="none"
              stroke="#FF66AA"
              strokeWidth="0.5"
              opacity="0.8"
            >
              <animate
                attributeName="r"
                values="4.5; 42; 25"
                dur="2.8s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0.8; 0; 0"
                dur="2.8s"
                repeatCount="indefinite"
              />
            </circle>
          </svg>
        )}

        {/* Hover Tooltip Overlay */}
        {hoveredMap && !selectedMap && (() => {
          const hoveredNode = polarNodes.find(n => n.map.map_hash === hoveredMap.map_hash);
          const angleDeg = hoveredNode ? ((hoveredNode.angle * 180 / Math.PI + 360) % 360).toFixed(0) : '0';
          const rDist = hoveredNode ? hoveredNode.r.toFixed(1) : '0.0';
          return (
            <div className="node-tooltip">
              <strong>#{polarNodes.findIndex(n => n.map.map_hash === hoveredMap.map_hash) + 1} {hoveredMap.title}</strong>
              <div className="tooltip-sub">{hoveredMap.artist} [{hoveredMap.difficulty}]</div>
              <div className="tooltip-sr" style={{ color: getSRColor(hoveredMap.sr || hoveredMap.star_rating) }}>
                ★ {(hoveredMap.sr || hoveredMap.star_rating || 0).toFixed(2)}
              </div>
              <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.7)', marginTop: '2px' }}>
                Distance: {rDist} u • Angle: {angleDeg}°
              </div>
            </div>
          );
        })()}
      </div>

      {/* Tabular Recommendation Ranking Widget directly under the graph */}
      {polarNodes.length > 0 && (
        <div className="fog-ranking-widget glass-card" style={{ marginTop: '20px', padding: '15px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h4 style={{ margin: 0, fontSize: '15px', color: 'var(--text-primary)' }}>
              📊 Recommended Maps Ranking Table
            </h4>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              Polar Skill Radar • Ordered by recommendation priority (#1 = top match)
            </span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="ranking-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-glass)', textAlign: 'left', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Rank</th>
                  <th style={{ padding: '8px' }}>Map Details</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Star Rating</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Match %</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Radar Pos (r, θ)</th>
                  <th style={{ padding: '8px' }}>Top 3 Influencing Played Maps</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {polarNodes.map((n, idx) => {
                  const m = n.map;
                  const sr = m.sr || m.star_rating || 0;
                  const diffColor = getSRColor(sr);
                  const isSelected = selectedMap && selectedMap.map_hash === m.map_hash;
                  const inflList = (m.influential_plays || (m.influential_play ? [m.influential_play] : [])).filter(Boolean).slice(0, 3);
                  const angleDeg = ((n.angle * 180 / Math.PI + 360) % 360).toFixed(0);

                  return (
                    <tr
                      key={m.map_hash || idx}
                      style={{
                        borderBottom: '1px solid rgba(255,255,255,0.05)',
                        backgroundColor: isSelected ? 'rgba(255, 102, 170, 0.15)' : 'transparent',
                        cursor: 'pointer'
                      }}
                      onClick={() => handleNodeClick(m)}
                    >
                      <td style={{ padding: '8px', textAlign: 'center', fontWeight: 'bold', color: idx === 0 ? '#00F3FF' : idx === 1 ? '#FFD700' : idx === 2 ? '#FF66AA' : 'var(--text-primary)' }}>
                        #{idx + 1}
                      </td>
                      <td style={{ padding: '8px' }}>
                        <div style={{ fontWeight: 'bold', color: 'var(--text-primary)' }}>{m.title}</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{m.artist} [{m.difficulty}] • {m.creator}</div>
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center' }}>
                        <span style={{ padding: '2px 6px', borderRadius: '4px', border: `1px solid ${diffColor}`, color: diffColor, fontWeight: 'bold' }}>
                          ★ {sr.toFixed(2)}
                        </span>
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center', color: 'var(--accent-neon-cyan)', fontWeight: 'bold' }}>
                        {((m.score || m.similarity || 0) * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                        {n.r.toFixed(1)} u, {angleDeg}°
                      </td>
                      <td style={{ padding: '8px' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          {inflList.length > 0 ? inflList.map((p, i) => (
                            <div key={i} style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '260px' }}>
                              #{i + 1} <strong style={{ color: 'var(--text-primary)' }}>{p.title}</strong> [{p.difficulty}] ({(p.similarity * 100).toFixed(0)}%)
                            </div>
                          )) : <span style={{ color: 'var(--text-secondary)' }}>N/A</span>}
                        </div>
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center' }}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleNodeClick(m);
                          }}
                          style={{
                            padding: '4px 10px',
                            borderRadius: '4px',
                            border: '1px solid var(--border-glass)',
                            background: 'rgba(255,255,255,0.08)',
                            color: 'var(--text-primary)',
                            fontSize: '11px',
                            cursor: 'pointer'
                          }}
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Selected Map Modal / Detail Sidecard */}
      {selectedMap && (() => {
        const selectedNode = polarNodes.find(n => n.map.map_hash === selectedMap.map_hash);
        const selAngleDeg = selectedNode ? ((selectedNode.angle * 180 / Math.PI + 360) % 360).toFixed(0) : '0';
        const selRDist = selectedNode ? selectedNode.r.toFixed(1) : '0.0';

        return (
          <div className="selected-map-modal-backdrop" onClick={() => setSelectedMap(null)}>
            <div className="selected-map-card glass-card" onClick={(e) => e.stopPropagation()}>
              <button className="close-btn" onClick={() => setSelectedMap(null)}>✕</button>
              <div className="map-card-header">
                <span className="card-badge" style={{ backgroundColor: getSRColor(selectedMap.sr || selectedMap.star_rating) }}>
                  ★ {(selectedMap.sr || selectedMap.star_rating || 0).toFixed(2)} SR
                </span>
                <h2>#{polarNodes.findIndex(n => n.map.map_hash === selectedMap.map_hash) + 1} {selectedMap.title}</h2>
                <h4>{selectedMap.artist}</h4>
              </div>

              <div className="map-card-body">
                <div className="stat-grid">
                  <div className="stat-box">
                    <span className="stat-label">Difficulty</span>
                    <span className="stat-value">{selectedMap.difficulty || 'Normal'}</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Mapper</span>
                    <span className="stat-value">{selectedMap.creator || 'Unknown'}</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Beatmap ID</span>
                    <span className="stat-value">{selectedMap.map_id || 'N/A'}</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Radar Pos (r, θ)</span>
                    <span className="stat-value">
                      {selRDist} units ({selAngleDeg}°)
                    </span>
                  </div>
                </div>

              {((selectedMap.influential_plays && selectedMap.influential_plays.length > 0) || selectedMap.influential_play) && (
                <div className="influential-play-section">
                  <div className="influential-title">
                    💡 <strong>Top 3 Played Maps Influencing This Recommendation:</strong>
                  </div>
                  <div className="influential-cards-list" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {(selectedMap.influential_plays || [selectedMap.influential_play]).slice(0, 3).map((play, idx) => (
                      <div key={idx} className="influential-card glass-card" style={{ padding: '8px 12px' }}>
                        <div className="influential-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span className="influential-map-title" style={{ fontWeight: 'bold' }}>#{idx + 1} {play.title}</span>
                          <span className="influential-diff" style={{ fontSize: '0.85em', opacity: 0.8 }}>[{play.difficulty}]</span>
                          {play.mods && (
                            <span className="mods-tag" style={{ fontSize: '0.7em', padding: '2px 6px', borderRadius: '4px', background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.15)' }}>
                              {getModsString(play.mods)}
                            </span>
                          )}
                        </div>
                        <div className="influential-sub" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px', fontSize: '0.85em' }}>
                          <span className="artist">{play.artist}</span>
                          <span className="match-badge">
                            ⚡ {(play.similarity * 100).toFixed(1)}% Match
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="map-card-actions" style={{ marginTop: '24px' }}>
              {selectedMap.map_id ? (
                <a
                  href={`https://osu.ppy.sh/b/${selectedMap.map_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="osu-link-button"
                >
                  <span className="btn-icon">🌐</span> View Map on osu!
                </a>
              ) : (
                <button disabled className="osu-link-button disabled">
                  Map ID unavailable
                </button>
              )}
            </div>
          </div>
        </div>
      );
    })()}
    </div>
  );
}

export default FogOfWarGraph;
