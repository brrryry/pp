import React, { useState, useEffect } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend
} from 'chart.js'
import { Scatter, Bar } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend
)

function ReplayModal({ play, onClose }) {
  const [hits, setHits] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchHits = async () => {
      if (!play) return
      setLoading(true)
      const basename = play.replay_file.replace('.osr', '')
      try {
        const res = await fetch(`/api/hits/${basename}`)
        if (res.ok) {
          const data = await res.json()
          setHits(data)
        }
      } catch (err) {
        console.error('Error fetching hits:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchHits()
  }, [play])

  if (!play) return null

  // 1. Prepare Aim Scatter Plot Data
  const scatterPoints = hits
    .filter(h => h.hit && h.dx !== null && h.dy !== null)
    .map(h => ({ x: h.dx, y: h.dy }))

  // Construct target ring (radius = 36px, representing CS4 outer border)
  const circlePoints = []
  for (let theta = 0; theta <= 2 * Math.PI + 0.1; theta += 0.1) {
    circlePoints.push({ x: 36 * Math.cos(theta), y: 36 * Math.sin(theta) })
  }

  const scatterData = {
    datasets: [
      {
        label: 'Hits Landing',
        data: scatterPoints,
        backgroundColor: 'rgba(325, 100%, 60%, 0.6)', // Neon Pink
        borderColor: 'hsl(325, 100%, 65%)',
        pointRadius: 3,
      },
      {
        label: 'CS4 Outer Edge',
        data: circlePoints,
        borderColor: 'rgba(220, 15%, 70%, 0.4)',
        borderWidth: 1.5,
        borderDash: [5, 5],
        fill: false,
        showLine: true,
        pointRadius: 0,
      }
    ]
  }

  const scatterOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        min: -50,
        max: 50,
        grid: { color: 'rgba(220, 20%, 30%, 0.15)' },
        title: { display: true, text: 'X Offset (pixels)', color: 'hsl(220, 15%, 70%)' },
        ticks: { color: 'hsl(220, 15%, 60%)' }
      },
      y: {
        min: -50,
        max: 50,
        grid: { color: 'rgba(220, 20%, 30%, 0.15)' },
        title: { display: true, text: 'Y Offset (pixels)', color: 'hsl(220, 15%, 70%)' },
        ticks: { color: 'hsl(220, 15%, 60%)' }
      }
    },
    plugins: {
      legend: { display: false }
    }
  }

  // 2. Prepare Timing Offset Histogram
  const timingOffsets = hits
    .filter(h => h.hit && h.timing_offset !== null)
    .map(h => h.timing_offset)

  const binSize = 10
  const minBin = -120
  const maxBin = 120
  const numBins = (maxBin - minBin) / binSize + 1

  const binCounts = new Array(numBins).fill(0)
  const timingLabels = []

  for (let i = 0; i < numBins; i++) {
    const binStart = minBin + i * binSize
    timingLabels.push(`${binStart}ms`)
  }

  timingOffsets.forEach(offset => {
    let binIdx = Math.floor((offset - minBin) / binSize)
    if (binIdx >= 0 && binIdx < numBins) {
      binCounts[binIdx]++
    }
  })

  const timingData = {
    labels: timingLabels,
    datasets: [
      {
        label: 'Hit Count',
        data: binCounts,
        backgroundColor: 'rgba(185, 100%, 50%, 0.6)', // Neon Cyan
        borderColor: 'hsl(185, 100%, 55%)',
        borderWidth: 1.5,
        barPercentage: 0.9,
        categoryPercentage: 0.9
      }
    ]
  }

  const timingOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        grid: { display: false },
        title: { display: true, text: 'Timing Error (Early < 0 < Late)', color: 'hsl(220, 15%, 70%)' },
        ticks: {
          color: 'hsl(220, 15%, 60%)',
          callback: function(val, index) {
            // Display every 3rd label
            return index % 3 === 0 ? this.getLabelForValue(val) : ''
          }
        }
      },
      y: {
        grid: { color: 'rgba(220, 20%, 30%, 0.15)' },
        title: { display: true, text: 'Notes Count', color: 'hsl(220, 15%, 70%)' },
        ticks: { color: 'hsl(220, 15%, 60%)' }
      }
    },
    plugins: {
      legend: { display: false }
    }
  }

  return (
    <div id="details-modal" className="modal" style={{ display: 'flex' }}>
      <div className="modal-content glass-card" style={{ animation: 'zoomIn 0.3s ease-out' }}>
        <header className="modal-header">
          <div>
            <h3 id="modal-map-title">{play.map_artist} - {play.map_title}</h3>
            <p id="modal-map-diff">Difficulty: {play.difficulty_name} | Player: {play.player}</p>
          </div>
          <span className="close-btn" id="close-modal-btn" onClick={onClose}>&times;</span>
        </header>
        
        {loading ? (
          <div className="modal-body" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '300px' }}>
            <span className="upload-icon" style={{ animation: 'spin 1s linear infinite', fontSize: '36px' }}>🔄</span>
            <p style={{ marginLeft: '12px' }}>Loading Diagnostics...</p>
          </div>
        ) : (
          <div className="modal-body">
            <div className="modal-chart-panel">
              <h4>Aim Landing Scatter (px relative to center)</h4>
              <div className="modal-chart-container">
                <Scatter data={scatterData} options={scatterOptions} />
              </div>
            </div>
            
            <div className="modal-chart-panel">
              <h4>Timing Offset Distribution (ms early/late)</h4>
              <div className="modal-chart-container2">
                <Bar data={timingData} options={timingOptions} />
              </div>
            </div>
          </div>
        )}
        
        <footer class="modal-footer">
          <div className="modal-stat">
            <span className="modal-stat-label">Accuracy:</span>
            <span className="modal-stat-value" id="modal-stat-acc">{play.accuracy_percent.toFixed(2)}%</span>
          </div>
          <div className="modal-stat">
            <span className="modal-stat-label">Unstable Rate:</span>
            <span className="modal-stat-value" id="modal-stat-ur">{play.unstable_rate.toFixed(1)}</span>
          </div>
          <div className="modal-stat">
            <span className="modal-stat-label">Aim Error:</span>
            <span className="modal-stat-value" id="modal-stat-aim">{play.avg_aim_error_px.toFixed(1)} px</span>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default ReplayModal
