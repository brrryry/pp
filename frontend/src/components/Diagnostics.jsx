import React from 'react'

function Diagnostics({ skills, summary }) {
  if (!skills || !summary) return null

  // Use mechanical skills first, fallback to potential if empty
  const activeSkills = Object.keys(skills.mechanical || {}).length > 0 
    ? skills.mechanical 
    : skills.potential || {}

  if (Object.keys(activeSkills).length === 0) {
    return (
      <div id="skill-description" className="skill-description">
        No play data available to generate diagnostics. Upload a replay to begin!
      </div>
    )
  }

  const weakSkills = []
  const strongSkills = []

  for (const [skill, val] of Object.entries(activeSkills)) {
    if (val < 65) weakSkills.push(skill)
    if (val >= 80) strongSkills.push(skill)
  }

  const avgUr = summary.avg_ur ? summary.avg_ur.toFixed(1) : '0.0'
  const avgAimError = summary.avg_aim_error ? summary.avg_aim_error.toFixed(1) : '0.0'

  const adviceList = []

  if (weakSkills.includes('Tech')) {
    adviceList.push(
      <li key="tech">
        <strong>Tech Aim Skill is low ({activeSkills.Tech?.toFixed(0)}):</strong> You struggle on irregular slider shapes and complex angles (e.g. Camellia tech maps). Practice on flow-aim beatmaps to learn smooth cursor transitions.
      </li>
    )
  }
  if (weakSkills.includes('Snap Aim')) {
    adviceList.push(
      <li key="snap">
        <strong>Snap Aim Skill is low ({activeSkills['Snap Aim']?.toFixed(0)}):</strong> Focus on linear jump maps with larger spacing to practice coordinate landing precision.
      </li>
    )
  }
  if (weakSkills.includes('Precision')) {
    adviceList.push(
      <li key="precision">
        <strong>Precision Skill is low ({activeSkills.Precision?.toFixed(0)}):</strong> Your timing (UR) is inconsistent. Try playing maps with higher Overall Difficulty (OD) at a lower Star Rating to build rhythmic accuracy.
      </li>
    )
  }
  if (weakSkills.includes('Speed')) {
    adviceList.push(
      <li key="speed">
        <strong>Speed Skill is low ({activeSkills.Speed?.toFixed(0)}):</strong> Focus on short burst maps to build tapping speed.
      </li>
    )
  }
  if (weakSkills.includes('Stamina')) {
    adviceList.push(
      <li key="stamina">
        <strong>Stamina Skill is low ({activeSkills.Stamina?.toFixed(0)}):</strong> Practice longer stream or high-BPM burst maps, building up from shorter maps.
      </li>
    )
  }
  if (weakSkills.includes('Streaming')) {
    adviceList.push(
      <li key="streaming">
        <strong>Streaming Skill is low ({activeSkills.Streaming?.toFixed(0)}):</strong> Rhythmic tapping consistency is low during fast streams. Practice on stream maps with lower OD to build muscle memory before raising OD.
      </li>
    )
  }
  if (weakSkills.includes('Reading')) {
    adviceList.push(
      <li key="reading">
        <strong>Reading Skill is low ({activeSkills.Reading?.toFixed(0)}):</strong> Visual patterns look cluttered. Try playing low AR (Approach Rate) maps with high density to learn note overlap parsing.
      </li>
    )
  }

  return (
    <div id="skill-description" className="skill-description">
      <p>
        Based on your plays, your timing precision has a mean Unstable Rate of <strong>{avgUr}</strong> and your average spatial aim drift is <strong>{avgAimError} px</strong>.
      </p>
      
      {strongSkills.length > 0 && (
        <p style={{ marginTop: '10px' }}>
          🏆 Your core strengths are: <strong>{strongSkills.join(', ')}</strong>.
        </p>
      )}

      {adviceList.length > 0 ? (
        <div className="skill-desc-list" style={{ marginTop: '15px' }}>
          <h4 style={{
            fontSize: '13px',
            fontWeight: '600',
            textTransform: 'uppercase',
            color: 'var(--accent-neon-pink)',
            letterSpacing: '0.5px',
            marginBottom: '5px'
          }}>
            Recommendations for Improvement:
          </h4>
          <ul>{adviceList}</ul>
        </div>
      ) : (
        <p style={{ marginTop: '15px', color: 'var(--accent-neon-emerald)' }}>
          ⭐ Your skills are highly balanced across all areas! Continue challenging yourself with harder map star ratings.
        </p>
      )}
    </div>
  )
}

export default Diagnostics
