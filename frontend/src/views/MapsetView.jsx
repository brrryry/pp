import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import ReplayModal from '../components/ReplayModal'
import { Radar } from 'react-chartjs-2'

const AXIS_ORDER = [
  'Snap Aim',
  'Speed',
  'Streaming',
  'Stamina',
  'Finger Control',
  'Reading',
  'Visual Density',
  'Tech',
  'Aim Control',
  'Flow Aim',
  'Precision'
]

const KEY_MAP = {
  'Snap Aim': 'SnapAim',
  'Speed': 'Speed',
  'Streaming': 'Streaming',
  'Stamina': 'Stamina',
  'Finger Control': 'FingerControl',
  'Reading': 'Reading',
  'Visual Density': 'VisualDensity',
  'Tech': 'Tech',
  'Aim Control': 'AimControl',
  'Flow Aim': 'FlowAim',
  'Precision': 'Precision'
}

function MapsetView() {
  const { mapset } = useParams() // The beatmap hash
  const navigate = useNavigate()
  const [mapData, setMapData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeReplay, setActiveReplay] = useState(null)

  // Prediction state
  const [users, setUsers] = useState([])
  const [selectedUser, setSelectedUser] = useState('')
  const [selectedMod, setSelectedMod] = useState('NoMod')
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)

  useEffect(() => {
    const fetchMapDetails = async () => {
      setLoading(true)
      try {
        const response = await fetch(`/api/map/${mapset}`)
        if (response.ok) {
          const data = await response.json()
          setMapData(data)
        } else {
          setMapData(null)
        }
      } catch (err) {
        console.error('Error fetching map details:', err)
        setMapData(null)
      } finally {
        setLoading(false)
      }
    }
    
    const fetchUsers = async () => {
      try {
        const response = await fetch('/api/users')
        if (response.ok) {
          const data = await response.json()
          setUsers(data)
          if (data.length > 0) {
            setSelectedUser(data[0])
          }
        }
      } catch (err) {
        console.error('Error fetching users:', err)
      }
    }

    fetchMapDetails()
    fetchUsers()
  }, [mapset])

  const handleOpenDiagnostics = (play) => {
    setActiveReplay(play)
  }

  const handleCloseDiagnostics = () => {
    setActiveReplay(null)
  }

  const handlePredict = async () => {
    if (!selectedUser) return
    setPredicting(true)
    setPrediction(null)
    try {
      const response = await fetch(`/api/predict?username=${selectedUser}&beatmap_hash=${mapset}&mods=${selectedMod}`)
      if (response.ok) {
        const data = await response.json()
        setPrediction(data.predicted_accuracy)
      } else {
        alert('Prediction failed. Make sure the predictor model is trained.')
      }
    } catch (err) {
      console.error('Prediction request error:', err)
    } finally {
      setPredicting(false)
    }
  }

  if (loading) {
    return (
      <div className="app-container">
        <Sidebar />
        <main className="main-content" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <span className="upload-icon" style={{ animation: 'spin 1s linear infinite', fontSize: '48px' }}>🔄</span>
          <h3 style={{ marginLeft: '12px' }}>Loading Beatmap Details...</h3>
        </main>
      </div>
    )
  }

  if (!mapData) {
    return (
      <div className="app-container">
        <Sidebar />
        <main className="main-content" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%', textAlign: 'center', padding: '20px' }}>
          <span style={{ fontSize: '48px', marginBottom: '15px' }}>⚠️</span>
          <h3>Beatmap Not Found</h3>
          <p className="subtext" style={{ marginTop: '8px' }}>
            We couldn't find map files or analyzed scores for the hash "{mapset}".
          </p>
          <button 
            className="view-diag-btn" 
            style={{ marginTop: '20px' }}
            onClick={() => navigate('/')}
          >
            Return Dashboard
          </button>
        </main>
      </div>
    )
  }

  // Map skills to ordering
  const skillsRaw = mapData.skills || {}
  const radarDataPoints = AXIS_ORDER.map(label => {
    const apiKey = KEY_MAP[label]
    return skillsRaw[apiKey] || 0
  })

  const radarData = {
    labels: AXIS_ORDER,
    datasets: [
      {
        label: 'Skillset Requirements',
        data: radarDataPoints,
        backgroundColor: 'rgba(270, 100%, 65%, 0.15)', // Neon Purple transparent
        borderColor: 'hsl(270, 100%, 65%)',
        borderWidth: 2,
        pointBackgroundColor: 'hsl(185, 100%, 50%)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'hsl(270, 100%, 65%)',
        pointRadius: 4,
      }
    ]
  }

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        angleLines: { color: 'rgba(220, 20%, 30%, 0.15)' },
        grid: { color: 'rgba(220, 20%, 30%, 0.15)' },
        pointLabels: {
          font: { size: 10, weight: '600', family: "'Outfit', sans-serif" },
          color: 'hsl(220, 30%, 95%)'
        },
        ticks: {
          backdropColor: 'transparent',
          color: 'hsl(220, 15%, 50%)',
          font: { size: 8 },
          stepSize: 20
        },
        min: 0,
        max: 100
      }
    },
    plugins: {
      legend: { display: false }
    }
  }

  return (
    <div className="app-container">
      <Sidebar />

      <main className="main-content">
        {/* Header */}
        <header className="profile-header">
          <div className="player-info" style={{ gap: '15px' }}>
            <span className="avatar" style={{ background: 'hsla(270, 100%, 65%, 0.15)', color: 'hsl(270, 100%, 70%)', filter: 'drop-shadow(0 0 5px var(--accent-neon-purple))' }}>🎵</span>
            <div>
              <h2>{mapData.title}</h2>
              <p className="subtext" style={{ marginTop: '2px', fontSize: '13px' }}>
                Artist: <strong>{mapData.artist}</strong> | Difficulty: <span className="mods-tag" style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-glass)' }}>{mapData.difficulty_name}</span> | Mapper: <strong>{mapData.creator}</strong>
              </p>
            </div>
          </div>
          <div className="quick-stats">
            <div className="stat-card">
              <span className="stat-value">{mapData.bpm}</span>
              <span className="stat-label">BPM</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">CS {mapData.cs}</span>
              <span className="stat-label">Circle Size</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">OD {mapData.od}</span>
              <span className="stat-label">Overall Diff</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">AR {mapData.ar}</span>
              <span className="stat-label">Approach Rate</span>
            </div>
          </div>
        </header>

        {/* Dashboard Grid */}
        <div className="dashboard-grid">
          {/* Radar Requirements Panel */}
          <section className="panel skill-chart-panel">
            <h3>Map Difficulty Demands</h3>
            <div className="chart-container" style={{ position: 'relative', height: '90%', minHeight: '300px' }}>
              <Radar data={radarData} options={radarOptions} />
            </div>
          </section>

          {/* Map metadata overview Panel */}
          <section className="panel skill-summary-panel" style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <h3>Map Info Summary</h3>
            <div className="skill-description" style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
              <p>
                This beatmap has a hash checksum of <code>{mapData.hash}</code>. It contains <strong>{mapData.cs ? 'Circle Size' : 'Hit Circles'}</strong> parameters optimized for aim coordination.
              </p>
              
              <div style={{ marginTop: '10px' }}>
                <h4 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--accent-neon-cyan)', marginBottom: '6px' }}>Parameter Breakdown:</h4>
                <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <li><strong>HP Drain:</strong> {mapData.hp}</li>
                  <li><strong>Timing Window (OD):</strong> Approximately ±{(80 - 6 * mapData.od).toFixed(1)}ms for 300 hits.</li>
                  <li><strong>Visual Preempt (AR):</strong> {(mapData.ar > 5 ? 1200 - 750 * (mapData.ar - 5) / 5 : 1200 + 600 * (5 - mapData.ar) / 5).toFixed(0)}ms render delay.</li>
                </ul>
              </div>

              <div style={{
                marginTop: 'auto',
                padding: '12px',
                borderRadius: '8px',
                background: 'hsla(185, 100%, 50%, 0.05)',
                border: '1px solid var(--border-glass)',
                fontSize: '12px',
                lineHeight: '1.4'
              }}>
                ⭐ <strong>Map Profiler Insight:</strong> Aim demands represent spatial movement limits, while Speed/Streaming requirements correspond to the tapping index.
              </div>
            </div>
          </section>
        </div>

        {/* Predictor Panel */}
        <section className="panel" style={{ marginTop: '24px', padding: '20px' }}>
          <h3>Performance Predictor (Accuracy Model)</h3>
          <p className="subtext" style={{ fontSize: '13px', marginBottom: '20px' }}>
            Estimate player accuracy on this map using historical user profiles and 11-axis skill mapping.
          </p>
          
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '30px', alignItems: 'center' }}>
            {/* Input Selection */}
            <div style={{ flex: '1', minWidth: '250px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>SELECT PLAYER</label>
                <select 
                  value={selectedUser} 
                  onChange={(e) => setSelectedUser(e.target.value)}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '8px',
                    background: 'var(--bg-glass)',
                    border: '1px solid var(--border-glass)',
                    color: 'var(--text-primary)',
                    fontFamily: 'inherit',
                    fontSize: '14px',
                    outline: 'none',
                    cursor: 'pointer'
                  }}
                >
                  {users.length === 0 ? (
                    <option value="">No players cached</option>
                  ) : (
                    users.map(u => (
                      <option key={u} value={u}>{u}</option>
                    ))
                  )}
                </select>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>SELECT MODS</label>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {['NoMod', 'HR', 'DT', 'EZ', 'HD', 'FL'].map(mod => (
                    <button
                      key={mod}
                      onClick={() => setSelectedMod(mod)}
                      style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        background: selectedMod === mod ? 'var(--accent-neon-cyan)' : 'var(--bg-glass)',
                        border: selectedMod === mod ? '1px solid var(--accent-neon-cyan)' : '1px solid var(--border-glass)',
                        color: selectedMod === mod ? '#000' : 'var(--text-primary)',
                      }}
                    >
                      {mod}
                    </button>
                  ))}
                </div>
              </div>

              <button
                className="view-diag-btn"
                onClick={handlePredict}
                disabled={predicting || !selectedUser}
                style={{
                  padding: '12px',
                  borderRadius: '8px',
                  fontWeight: '600',
                  marginTop: '10px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                {predicting ? 'Computing Predictions...' : 'Run Performance Predictor ⚡'}
              </button>
            </div>

            {/* Prediction Output Visualizer */}
            <div style={{
              flex: '1',
              minWidth: '250px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '20px',
              borderRadius: '12px',
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px dashed var(--border-glass)',
              minHeight: '180px',
              textAlign: 'center'
            }}>
              {prediction !== null ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                  <div style={{
                    width: '100px',
                    height: '100px',
                    borderRadius: '50%',
                    border: '4px solid var(--accent-neon-cyan)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '20px',
                    fontWeight: '800',
                    color: 'var(--text-primary)',
                    boxShadow: '0 0 15px rgba(0, 243, 255, 0.3)'
                  }}>
                    {prediction.toFixed(2)}%
                  </div>
                  <h4 style={{ color: 'var(--text-primary)', marginTop: '8px' }}>Predicted Accuracy</h4>
                  <p style={{
                    fontSize: '12px',
                    color: prediction >= 98.0 ? 'var(--accent-neon-emerald)' :
                           prediction >= 95.0 ? 'var(--accent-neon-cyan)' :
                           prediction >= 90.0 ? '#ffb700' : 'var(--accent-neon-pink)',
                    fontWeight: '600',
                    maxWidth: '300px',
                    marginTop: '4px'
                  }}>
                    {prediction >= 98.0 ? '🏆 SS Rank Candidate. Highly favorable pattern matching!' :
                     prediction >= 95.0 ? '⭐ High Performance. Stable tap sync predicted.' :
                     prediction >= 90.0 ? '⚠️ Pass Candidate. Circle size/spacing pushes execution limits.' :
                     '🛑 High Failure Risk. Tapping speeds exceed historical stamina thresholds.'}
                  </p>
                </div>
              ) : (
                <div style={{ color: 'var(--text-secondary)' }}>
                  <span style={{ fontSize: '32px', display: 'block', marginBottom: '8px' }}>🤖</span>
                  <p style={{ fontSize: '13px' }}>Select a player and click predict to run the ML inference engine.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Local Leaderboard Panel */}
        <section className="panel table-panel">
          <h3>Local Scores Leaderboard</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Accuracy</th>
                  <th>Unstable Rate</th>
                  <th>Aim Error (px)</th>
                  <th>Misses</th>
                  <th>Mods</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {mapData.leaderboard.length === 0 ? (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '20px', color: 'var(--text-secondary)' }}>
                      No local analyzed replays for this map. Upload a replay to show scores!
                    </td>
                  </tr>
                ) : (
                  mapData.leaderboard.map((score, idx) => {
                    const accPercent = Number(score.accuracy) || 0
                    const accClass = accPercent >= 99.0 ? 'acc-good' : (accPercent < 90.0 ? 'acc-bad' : '')
                    const ur = Number(score.unstable_rate) || 0
                    const urClass = ur < 150.0 ? 'ur-low' : ''
                    
                    return (
                      <tr key={idx}>
                        <td>
                          <strong>{score.player}</strong>
                        </td>
                        <td className={accClass}>{accPercent.toFixed(2)}%</td>
                        <td className={urClass}>{ur.toFixed(1)}</td>
                        <td>{Number(score.avg_aim_error).toFixed(1)}</td>
                        <td style={{ color: score.misses > 0 ? 'var(--accent-neon-pink)' : 'var(--accent-neon-emerald)' }}>
                          {score.misses}
                        </td>
                        <td>
                          <span className="mods-tag">{score.mods || 'NoMod'}</span>
                        </td>
                        <td>
                          <button
                            className="view-diag-btn"
                            onClick={() => handleOpenDiagnostics({
                              replay_file: score.replay_file,
                              player: score.player,
                              map_title: mapData.title,
                              map_artist: mapData.artist,
                              difficulty_name: mapData.difficulty_name,
                              accuracy_percent: score.accuracy,
                              unstable_rate: score.unstable_rate,
                              avg_aim_error_px: score.avg_aim_error
                            })}
                          >
                            Diagnostics
                          </button>
                        </td>
                      </tr>
                    )
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      {/* Diagnostics Modal */}
      {activeReplay && (
        <ReplayModal
          play={activeReplay}
          onClose={handleCloseDiagnostics}
        />
      )}
    </div>
  )
}

export default MapsetView
