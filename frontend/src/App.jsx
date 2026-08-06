import React from 'react'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './views/Dashboard'
import PlayerProfile from './views/PlayerProfile'
import MapsetView from './views/MapsetView'

function App() {
  return (
    <HashRouter>
      <div className="glass-bg"></div>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/player/:player" element={<PlayerProfile />} />
        <Route path="/mapset/:mapset" element={<MapsetView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  )
}

export default App
