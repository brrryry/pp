import React, { useState, useEffect, useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

function Sidebar() {
  const [users, setUsers] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadFileName, setUploadFileName] = useState('')
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)
  
  const navigate = useNavigate()
  const { player } = useParams()

  const filteredUsers = searchQuery.trim()
    ? users.filter((username) =>
        username.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : []

  const displayedUsers = filteredUsers.slice(0, 10)
  const remainingCount = filteredUsers.length - displayedUsers.length

  const loadUsers = async () => {
    try {
      const response = await fetch('/api/users')
      if (response.ok) {
        const data = await response.json()
        setUsers(data)
      }
    } catch (err) {
      console.error('Error loading users:', err)
    }
  }

  useEffect(() => {
    loadUsers()
  }, [])

  const handleUserClick = (username) => {
    navigate(`/player/${username}`)
  }

  const handleUpload = async (file) => {
    if (!file) return
    setUploading(true)
    setUploadFileName(file.name)
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Server analysis error')
      }
      
      const result = await response.json()
      const newPlayer = result.analysis.player
      
      // Reload user list and navigate to new player profile
      await loadUsers()
      navigate(`/player/${newPlayer}`)
      setSearchQuery('')
    } catch (err) {
      alert(`Analysis Failed: ${err.message}`)
      console.error(err)
    } finally {
      setUploading(false)
      setUploadFileName('')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const onDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const onDragLeave = () => {
    setIsDragOver(false)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files[0])
    }
  }

  return (
    <aside className="sidebar">
      <div className="logo" style={{ cursor: 'pointer' }} onClick={() => navigate('/')}>
        <span className="logo-icon">🎯</span>
        <h1>Osu! Profiler</h1>
      </div>
      
      <div className="user-section">
        <h2>Profiles</h2>
        <div className="search-bar-container">
          <input
            type="text"
            className="search-input"
            placeholder="Search players..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button className="search-clear-btn" onClick={() => setSearchQuery('')}>
              ✕
            </button>
          )}
        </div>
        <ul id="user-list">
          {users.length === 0 ? (
            <li className="subtext" style={{ padding: '10px' }}>
              No profiles yet. Upload a replay to begin!
            </li>
          ) : searchQuery.trim() === '' ? (
            player ? (
              <li
                key={player}
                className="user-item active"
                onClick={() => handleUserClick(player)}
              >
                👤 <span style={{ marginLeft: '8px' }}>{player}</span>
              </li>
            ) : (
              <li className="subtext" style={{ padding: '10px', fontSize: '12px' }}>
                Search above to find a player profile.
              </li>
            )
          ) : (
            <>
              {displayedUsers.map((username) => (
                <li
                  key={username}
                  className={`user-item ${player === username ? 'active' : ''}`}
                  onClick={() => handleUserClick(username)}
                >
                  👤 <span style={{ marginLeft: '8px' }}>{username}</span>
                </li>
              ))}
              {remainingCount > 0 && (
                <li className="remaining-matches-indicator">
                  + {remainingCount} more matches. Keep typing to filter...
                </li>
              )}
              {filteredUsers.length === 0 && (
                <li className="subtext" style={{ padding: '10px' }}>
                  No matching profiles found
                </li>
              )}
            </>
          )}
        </ul>
      </div>
      
      <div className="upload-section">
        <h2>Analyze Replay</h2>
        <div 
          id="drop-zone" 
          className={`drop-zone ${isDragOver ? 'dragover' : ''}`}
          onClick={() => fileInputRef.current.click()}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          {uploading ? (
            <>
              <span className="upload-icon" style={{ display: 'inline-block', animation: 'spin 1s linear infinite' }}>🔄</span>
              <p>Analyzing <strong>{uploadFileName}</strong>...</p>
              <p className="subtext">Downloading map if missing</p>
            </>
          ) : (
            <>
              <span className="upload-icon">📥</span>
              <p>Drag & Drop <strong>.osr</strong> here</p>
              <p class="subtext" style={{ margin: '5px 0 0 0' }}>or click to browse</p>
            </>
          )}
          <input 
            type="file" 
            ref={fileInputRef}
            accept=".osr" 
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files.length > 0) {
                handleUpload(e.target.files[0])
              }
            }}
          />
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
