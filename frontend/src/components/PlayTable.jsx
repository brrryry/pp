import React, { useState } from 'react'

function PlayTable({ plays = [], topPlays = [], recentPlays = [], onViewDiagnostics }) {
  const [activeTab, setActiveTab] = useState('local') // 'local', 'top', 'recent'

  const getActiveData = () => {
    switch (activeTab) {
      case 'top':
        return topPlays
      case 'recent':
        return recentPlays
      case 'local':
      default:
        // Reverse local plays so recent analysis is first
        return [...plays].reverse()
    }
  }

  const activeData = getActiveData()

  return (
    <section className="panel table-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', flexWrap: 'wrap', gap: '10px' }}>
        <h3>Play History</h3>
        
        {/* Tab Buttons */}
        <div style={{ display: 'flex', gap: '5px' }}>
          <button
            className={`view-diag-btn ${activeTab === 'local' ? '' : 'inactive'}`}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: activeTab === 'local' ? '1px solid var(--accent-neon-cyan)' : '1px solid var(--border-glass)',
              background: activeTab === 'local' ? 'hsla(185, 100%, 50%, 0.15)' : 'transparent',
              color: activeTab === 'local' ? 'hsl(185, 100%, 55%)' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}
            onClick={() => setActiveTab('local')}
          >
            Local ({plays.length})
          </button>
          
          <button
            className={`view-diag-btn ${activeTab === 'top' ? '' : 'inactive'}`}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: activeTab === 'top' ? '1px solid var(--accent-neon-cyan)' : '1px solid var(--border-glass)',
              background: activeTab === 'top' ? 'hsla(185, 100%, 50%, 0.15)' : 'transparent',
              color: activeTab === 'top' ? 'hsl(185, 100%, 55%)' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}
            onClick={() => setActiveTab('top')}
          >
            Top 200 ({topPlays.length})
          </button>

          <button
            className={`view-diag-btn ${activeTab === 'recent' ? '' : 'inactive'}`}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: activeTab === 'recent' ? '1px solid var(--accent-neon-cyan)' : '1px solid var(--border-glass)',
              background: activeTab === 'recent' ? 'hsla(185, 100%, 50%, 0.15)' : 'transparent',
              color: activeTab === 'recent' ? 'hsl(185, 100%, 55%)' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}
            onClick={() => setActiveTab('recent')}
          >
            Recent 50 ({recentPlays.length})
          </button>
        </div>
      </div>

      <div className="table-scroll" style={{ overflowY: 'auto', maxHeight: '400px' }}>
        <table>
          <thead>
            <tr>
              <th>Beatmap Title</th>
              <th>Difficulty</th>
              <th>Accuracy</th>
              <th>PP</th>
              <th>Unstable Rate</th>
              <th>Aim Error (px)</th>
              <th>Mods</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {activeData.length === 0 ? (
              <tr>
                <td colSpan="8" style={{ textAlign: 'center', padding: '30px', color: 'var(--text-secondary)' }}>
                  No plays found for this category.
                </td>
              </tr>
            ) : (
              activeData.map((play, idx) => {
                const accPercent = Number(play.accuracy_percent) || 0
                const accClass = accPercent >= 99.0 ? 'acc-good' : (accPercent < 90.0 ? 'acc-bad' : '')
                
                const ur = play.unstable_rate !== null && play.unstable_rate !== undefined ? Number(play.unstable_rate) : null
                const urClass = ur !== null && ur < 150.0 ? 'ur-low' : ''
                
                let mapTitle = `${play.map_artist} - ${play.map_title}`
                const fullTitle = mapTitle
                if (mapTitle.length > 45) {
                  mapTitle = mapTitle.substring(0, 42) + '...'
                }

                return (
                  <tr key={idx}>
                    <td title={fullTitle}>
                      <strong>{mapTitle}</strong>
                    </td>
                    <td>
                      <span className="subtext">{play.difficulty_name}</span>
                    </td>
                    <td className={accClass}>
                      {accPercent.toFixed(2)}%
                    </td>
                    <td>
                      {play.pp !== null && play.pp !== undefined ? (
                        <span style={{ fontWeight: 600, color: 'hsl(185, 100%, 55%)' }}>
                          {Math.round(play.pp)}pp
                        </span>
                      ) : (
                        <span className="subtext">-</span>
                      )}
                    </td>
                    <td className={urClass}>
                      {ur !== null ? ur.toFixed(1) : '-'}
                    </td>
                    <td>
                      {play.avg_aim_error_px !== null && play.avg_aim_error_px !== undefined 
                        ? Number(play.avg_aim_error_px).toFixed(1) 
                        : '-'}
                    </td>
                    <td>
                      <span className="mods-tag">{play.mods || 'NoMod'}</span>
                    </td>
                    <td>
                      {play.replay_file ? (
                        <button
                          className="view-diag-btn"
                          onClick={() => onViewDiagnostics(play)}
                        >
                          View
                        </button>
                      ) : play.beatmap_hash ? (
                        <button
                          className="view-diag-btn"
                          style={{
                            borderColor: 'var(--accent-neon-purple)',
                            color: 'hsl(270, 100%, 75%)',
                            background: 'hsla(270, 100%, 65%, 0.1)'
                          }}
                          onClick={() => {
                            // Navigate to beatmap detail page!
                            window.location.hash = `/mapset/${play.beatmap_hash}`
                          }}
                        >
                          Map Stats
                        </button>
                      ) : (
                        <span className="subtext" style={{ fontSize: '10px' }}>API score</span>
                      )}
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default PlayTable
