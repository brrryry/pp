import React, { useState } from 'react'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'
import { Radar } from 'react-chartjs-2'

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
)

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

function ProfileRadar({ skills }) {
  const [relative, setRelative] = useState(false)
  const [showPotential, setShowPotential] = useState(true)
  const [showMechanical, setShowMechanical] = useState(true)

  if (!skills) return <div className="subtext">No skill data loaded.</div>

  const potentialRaw = skills.potential || {}
  const mechanicalRaw = skills.mechanical || {}

  // Helper to normalize values
  const getNormalized = (rawDict) => {
    const vals = Object.values(rawDict).map(v => Number(v) || 0)
    const maxVal = Math.max(...vals, 1)
    const normalized = {}
    Object.keys(rawDict).forEach(k => {
      normalized[k] = Math.round(((Number(rawDict[k]) || 0) / maxVal) * 1000) / 10
    })
    return normalized
  }

  const potentialNorm = relative ? getNormalized(potentialRaw) : potentialRaw
  const mechanicalNorm = relative ? getNormalized(mechanicalRaw) : mechanicalRaw

  // Map to structured arrays in specific order
  const potentialData = AXIS_ORDER.map(label => potentialNorm[label] || 0)
  const mechanicalData = AXIS_ORDER.map(label => mechanicalNorm[label] || 0)

  const datasets = []

  if (showPotential && Object.keys(potentialRaw).length > 0) {
    datasets.push({
      label: 'Potential (Map Requirements)',
      data: potentialData,
      backgroundColor: 'rgba(185, 100%, 50%, 0.15)', // Neon Cyan transparent
      borderColor: 'hsl(185, 100%, 50%)',
      borderWidth: 2,
      pointBackgroundColor: 'hsl(185, 100%, 55%)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'hsl(185, 100%, 50%)',
      pointRadius: 4,
    })
  }

  if (showMechanical && Object.keys(mechanicalRaw).length > 0) {
    datasets.push({
      label: 'Mechanical (Your Execution)',
      data: mechanicalData,
      backgroundColor: 'rgba(325, 100%, 60%, 0.15)', // Neon Pink transparent
      borderColor: 'hsl(325, 100%, 60%)',
      borderWidth: 2,
      pointBackgroundColor: 'hsl(325, 100%, 65%)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'hsl(325, 100%, 60%)',
      pointRadius: 4,
    })
  }

  const data = {
    labels: AXIS_ORDER,
    datasets: datasets
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        angleLines: {
          color: 'rgba(220, 20%, 30%, 0.15)'
        },
        grid: {
          color: 'rgba(220, 20%, 30%, 0.15)'
        },
        pointLabels: {
          font: {
            size: 11,
            weight: '600',
            family: "'Outfit', sans-serif"
          },
          color: 'hsl(220, 30%, 95%)'
        },
        ticks: {
          backdropColor: 'transparent',
          color: 'hsl(220, 15%, 50%)',
          font: { size: 9 },
          stepSize: 20
        },
        min: 0,
        max: 100
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: 'hsl(220, 15%, 70%)',
          font: {
            family: "'Outfit', sans-serif",
            size: 11
          }
        }
      }
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Interactive Toggles */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', gap: '15px' }}>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <input
              type="checkbox"
              checked={showPotential}
              onChange={(e) => setShowPotential(e.target.checked)}
              style={{ marginRight: '6px' }}
            />
            Potential
          </label>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <input
              type="checkbox"
              checked={showMechanical}
              onChange={(e) => setShowMechanical(e.target.checked)}
              style={{ marginRight: '6px' }}
            />
            Mechanical
          </label>
        </div>

        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px', color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={relative}
            onChange={(e) => setRelative(e.target.checked)}
            style={{ marginRight: '6px' }}
          />
          Relative Scale (Normalize)
        </label>
      </div>

      <div style={{ flex: 1, position: 'relative', minHeight: '300px' }}>
        <Radar data={data} options={options} />
      </div>
    </div>
  )
}

export default ProfileRadar
