import React, { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import ProfileRadar from '../components/ProfileRadar'
import Diagnostics from '../components/Diagnostics'
import PlayTable from '../components/PlayTable'
import ReplayModal from '../components/ReplayModal'

function PlayerProfile() {
  const { player } = useParams()
  const [profileData, setProfileData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeReplay, setActiveReplay] = useState(null)

  useEffect(() => {
    const fetchProfile = async () => {
      setLoading(true)
      try {
        const response = await fetch(`/api/user/${player}`)
        if (response.ok) {
          const data = await response.json()
          setProfileData(data)
        } else {
          setProfileData(null)
        }
      } catch (err) {
        console.error('Error fetching player profile:', err)
        setProfileData(null)
      } finally {
        setLoading(false)
      }
    }
    fetchProfile()
  }, [player])

  const handleOpenDiagnostics = (play) => {
    setActiveReplay(play)
  }

  const handleCloseDiagnostics = () => {
    setActiveReplay(null)
  }

  return (
    <div className="app-container">
      <Sidebar />

      {/* Main Content Area */}
      <main className="main-content">
        {loading ? (
          <div style={{ display: 'flex', flex: '1', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <span className="upload-icon" style={{ animation: 'spin 1s linear infinite', fontSize: '48px' }}>🔄</span>
            <h3 style={{ marginTop: '15px' }}>Loading {player}'s Profile...</h3>
          </div>
        ) : !profileData ? (
          <div style={{ display: 'flex', flex: '1', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%', textAlign: 'center', padding: '20px' }}>
            <span style={{ fontSize: '48px', marginBottom: '15px' }}>⚠️</span>
            <h3>Profile Not Found</h3>
            <p className="subtext" style={{ marginTop: '8px' }}>
              We couldn't find any local replays or API profile logs for "{player}".
            </p>
          </div>
        ) : (
          <>
            {/* Profile Header */}
            <header className="profile-header">
              <div className="player-info">
                <span className="avatar">👤</span>
                <div>
                  <h2 id="active-username">{profileData.username}</h2>
                  <p id="player-subtitle">Skill Profile Dashboard | Mods: NoMod/Modded</p>
                </div>
              </div>
              <div className="quick-stats">
                <div className="stat-card">
                  <span className="stat-value" id="stat-total-plays">{profileData.summary.total_plays}</span>
                  <span className="stat-label">Total Plays</span>
                </div>
                <div className="stat-card">
                  <span className="stat-value" id="stat-avg-acc">{profileData.summary.avg_accuracy.toFixed(2)}%</span>
                  <span className="stat-label">Avg Accuracy</span>
                </div>
                <div className="stat-card font-neon">
                  <span className="stat-value" id="stat-avg-ur">{profileData.summary.avg_ur.toFixed(1)}</span>
                  <span className="stat-label">Avg UR</span>
                </div>
                <div className="stat-card">
                  <span className="stat-value" id="stat-avg-aim">{profileData.summary.avg_aim_error.toFixed(1)} px</span>
                  <span className="stat-label">Avg Aim Error</span>
                </div>
              </div>
            </header>

            {/* Dashboard Grid */}
            <div className="dashboard-grid">
              {/* Radar Chart Panel */}
              <section className="panel skill-chart-panel">
                <h3>Personalized Skill Profiling</h3>
                <div className="chart-container" style={{ position: 'relative', height: '90%' }}>
                  <ProfileRadar skills={profileData.skills} />
                </div>
              </section>

              {/* Suggestions / Diagnostics Panel */}
              <section className="panel skill-summary-panel">
                <h3>Performance Diagnostics</h3>
                <Diagnostics skills={profileData.skills} summary={profileData.summary} />
              </section>
            </div>

            {/* Play History Table Panel */}
            <PlayTable
              plays={profileData.plays}
              topPlays={profileData.top_plays}
              recentPlays={profileData.recent_plays}
              onViewDiagnostics={handleOpenDiagnostics}
            />
          </>
        )}
      </main>

      {/* Diagnostics Replay Modal */}
      {activeReplay && (
        <ReplayModal
          play={activeReplay}
          onClose={handleCloseDiagnostics}
        />
      )}
    </div>
  )
}

export default PlayerProfile
