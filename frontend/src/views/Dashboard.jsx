import React from 'react'
import Sidebar from '../components/Sidebar'

function Dashboard() {
  return (
    <div className="app-container">
      <Sidebar />
      
      <main className="main-content" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', padding: '40px' }}>
        <div style={{
          padding: '40px',
          borderRadius: '16px',
          background: 'hsla(225, 20%, 8%, 0.3)',
          border: '1px solid var(--border-glass)',
          maxWidth: '550px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
          backdropFilter: 'blur(10px)'
        }}>
          <span style={{ fontSize: '48px', display: 'block', marginBottom: '20px', filter: 'drop-shadow(0 0 10px var(--accent-neon-cyan))' }}>👋</span>
          <h2 style={{ fontSize: '24px', fontWeight: 800, marginBottom: '15px', color: '#fff' }}>Welcome to Osu! Profiler</h2>
          <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6, fontSize: '14px', marginBottom: '25px' }}>
            Select an existing player profile from the sidebar to inspect detailed 11-axis skill sets, execution timing diagnostics, and recommendations.
          </p>
          <div style={{
            padding: '15px',
            borderRadius: '10px',
            border: '1px dashed var(--accent-neon-pink)',
            background: 'hsla(325, 100%, 60%, 0.05)',
            color: 'hsl(325, 100%, 75%)',
            fontSize: '13px',
            fontWeight: 600
          }}>
            💡 Or drag and drop any .osr replay file into the sidebar to upload and instantly map performance!
          </div>
        </div>
      </main>
    </div>
  )
}

export default Dashboard
