import React, { useState, useEffect } from 'react'

function RecommendationsTab({ player }) {
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [limit, setLimit] = useState(10)

  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true)
      setError(null)
      try {
        const response = await fetch(`/api/recommend/${player}?limit=${limit}`)
        if (response.ok) {
          const data = await response.json()
          setRecommendations(data.recommendations || [])
        } else {
          const errData = await response.json()
          setError(errData.detail || 'Failed to fetch recommendations.')
        }
      } catch (err) {
        console.error('Error fetching recommendations:', err)
        setError('Network error loading recommendations.')
      } finally {
        setLoading(false)
      }
    }

    if (player) {
      fetchRecommendations()
    }
  }, [player, limit])

  const filteredRecs = recommendations.filter((r) => {
    const search = searchTerm.toLowerCase()
    return (
      r.title.toLowerCase().includes(search) ||
      r.artist.toLowerCase().includes(search) ||
      r.version.toLowerCase().includes(search) ||
      r.creator.toLowerCase().includes(search)
    );
  })

  const getDifficultyColor = (sr) => {
    if (sr < 2.0) return 'hsl(120, 100%, 60%)' // Easy - Green
    if (sr < 3.5) return 'hsl(190, 100%, 50%)' // Normal - Blue
    if (sr < 5.0) return 'hsl(50, 100%, 55%)'  // Hard - Yellow/Orange
    if (sr < 6.5) return 'hsl(330, 100%, 60%)' // Insane/Expert - Pink/Red
    return 'hsl(270, 100%, 65%)'               // Expert+ - Purple
  }

  const handleRowClick = (mapHash) => {
    window.location.hash = `/mapset/${mapHash}`
  }

  return (
    <section className="panel table-panel" style={{ flex: '1', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <h3>Map Recommendations</h3>
          <p className="subtext" style={{ marginTop: '2px' }}>
            Recommended maps tailored to match your specific skill capabilities.
          </p>
        </div>

        {/* Filter Controls */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="🔍 Filter recommendations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid var(--border-glass)',
              background: 'rgba(0, 0, 0, 0.2)',
              color: 'var(--text-primary)',
              fontSize: '13px',
              outline: 'none',
              width: '200px'
            }}
          />

          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid var(--border-glass)',
              background: 'rgba(0, 0, 0, 0.2)',
              color: 'var(--text-primary)',
              fontSize: '13px',
              outline: 'none',
              cursor: 'pointer'
            }}
          >
            <option value={20}>20 Maps</option>
            <option value={50}>50 Maps</option>
            <option value={100}>100 Maps</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', flex: '1' }}>
          <span className="upload-icon" style={{ animation: 'spin 1s linear infinite', fontSize: '32px' }}>🔄</span>
          <h4 style={{ marginTop: '10px', color: 'var(--text-secondary)' }}>Calculating best maps for {player}...</h4>
        </div>
      ) : error ? (
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', flex: '1', textAlign: 'center' }}>
          <span style={{ fontSize: '32px', marginBottom: '10px' }}>⚠️</span>
          <h4 style={{ color: 'var(--accent-neon-pink)' }}>Error Loading Recommendations</h4>
          <p className="subtext" style={{ marginTop: '5px' }}>{error}</p>
          <p className="subtext" style={{ marginTop: '10px', fontSize: '12px' }}>
            Hint: Make sure you have at least some beatmaps ingested and the ML model trained (`train_predictor.py`).
          </p>
        </div>
      ) : filteredRecs.length === 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', flex: '1', textAlign: 'center' }}>
          <span style={{ fontSize: '32px', marginBottom: '10px' }}>🗺️</span>
          <h4>No Recommendations Found</h4>
          <p className="subtext" style={{ marginTop: '5px' }}>
            We couldn't find any maps in the database matching your Star Rating range.
          </p>
        </div>
      ) : (
        <div className="table-scroll" style={{ overflowY: 'auto', maxHeight: '500px', flex: '1' }}>
          <table>
            <thead>
              <tr>
                <th>Map Details</th>
                <th>Difficulty Name</th>
                <th style={{ textAlign: 'center' }}>Star Rating</th>
                <th style={{ textAlign: 'center' }}>Predicted Acc</th>
                <th style={{ textAlign: 'center' }}>Similarity Match</th>
                <th>Top Influencing Plays</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecs.map((rec) => {
                const sr = rec.star_rating || rec.sr || 0
                const diffColor = getDifficultyColor(sr)
                const accPercent = rec.predicted_accuracy || 0
                const accClass = accPercent >= 97.0 ? 'acc-good' : (accPercent < 90.0 ? 'acc-bad' : '')
                const inflList = (rec.influential_plays || (rec.influential_play ? [rec.influential_play] : [])).filter(Boolean).slice(0, 3)

                let title = `${rec.artist || 'Unknown'} - ${rec.title || 'Unknown'}`
                const fullTitle = title
                if (title.length > 50) {
                  title = title.substring(0, 47) + '...'
                }

                return (
                  <tr
                    key={rec.map_hash}
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleRowClick(rec.map_hash)}
                  >
                    <td title={fullTitle}>
                      <strong>{title}</strong>
                      <div className="subtext" style={{ fontSize: '11px', marginTop: '2px' }}>Mapped by {rec.creator || 'Unknown'}</div>
                    </td>
                    <td>
                      <span className="subtext">{rec.version || rec.difficulty || 'Normal'}</span>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <span
                        style={{
                          display: 'inline-block',
                          padding: '3px 8px',
                          borderRadius: '4px',
                          background: 'rgba(0, 0, 0, 0.3)',
                          border: `1px solid ${diffColor}`,
                          color: diffColor,
                          fontWeight: 'bold',
                          fontSize: '12px'
                        }}
                      >
                        {sr.toFixed(2)} ★
                      </span>
                    </td>
                    <td className={accClass} style={{ textAlign: 'center', fontWeight: '600' }}>
                      {accPercent.toFixed(2)}%
                    </td>
                    <td style={{ textAlign: 'center', color: 'var(--accent-neon-cyan)', fontWeight: '500' }}>
                      {((rec.similarity || 0) * 100.0).toFixed(1)}%
                    </td>
                    <td>
                      <div style={{ fontSize: '11px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        {inflList.length > 0 ? inflList.map((p, i) => (
                          <div key={i} style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '240px' }}>
                            #{i+1} <strong style={{ color: 'var(--text-primary)' }}>{p.title}</strong> [{p.difficulty}] ({(p.similarity * 100).toFixed(0)}%)
                          </div>
                        )) : <span className="subtext">N/A</span>}
                      </div>
                    </td>
                    <td>
                      <button
                        className="view-diag-btn"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleRowClick(rec.map_hash)
                        }}
                        style={{
                          padding: '4px 10px',
                          fontSize: '11px',
                          borderRadius: '4px',
                          border: '1px solid var(--border-glass)',
                          background: 'rgba(255, 255, 255, 0.05)',
                          color: 'var(--text-primary)',
                          cursor: 'pointer'
                        }}
                      >
                        Inspect Map
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default RecommendationsTab
