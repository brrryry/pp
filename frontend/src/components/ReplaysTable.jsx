import React, { useState, useMemo } from 'react';

// Helper function to convert numeric bitwise mods to standard strings
function parseMods(modsInt) {
  if (!modsInt || modsInt === 0) return 'NM';
  const MOD_FLAGS = [
    { flag: 1 << 0, name: 'NF' },
    { flag: 1 << 1, name: 'EZ' },
    { flag: 1 << 2, name: 'TD' },
    { flag: 1 << 3, name: 'HD' },
    { flag: 1 << 4, name: 'HR' },
    { flag: 1 << 5, name: 'SD' },
    { flag: 1 << 6, name: 'DT' },
    { flag: 1 << 7, name: 'RX' },
    { flag: 1 << 8, name: 'HT' },
    { flag: 1 << 9, name: 'NC' },
    { flag: 1 << 10, name: 'FL' },
    { flag: 1 << 12, name: 'SO' },
    { flag: 1 << 14, name: 'PF' },
  ];

  let result = '';
  for (const mod of MOD_FLAGS) {
    if ((modsInt & mod.flag) !== 0) {
      result += mod.name;
    }
  }
  return result || 'NM';
}

function ReplaysTable({ replays = [] }) {
  const [filterText, setFilterText] = useState('');
  const [sortField, setSortField] = useState('mastery_score');
  const [sortDir, setSortDir] = useState('desc');

  const filteredAndSortedReplays = useMemo(() => {
    let list = [...replays];

    if (filterText.trim()) {
      const q = filterText.toLowerCase();
      list = list.filter(r =>
        (r.title && r.title.toLowerCase().includes(q)) ||
        (r.artist && r.artist.toLowerCase().includes(q)) ||
        (r.difficulty && r.difficulty.toLowerCase().includes(q)) ||
        (r.creator && r.creator.toLowerCase().includes(q))
      );
    }

    list.sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      if (typeof valA === 'string') valA = valA.toLowerCase();
      if (typeof valB === 'string') valB = valB.toLowerCase();

      if (valA < valB) return sortDir === 'asc' ? -1 : 1;
      if (valA > valB) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });

    return list;
  }, [replays, filterText, sortField, sortDir]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  const renderSortIndicator = (field) => {
    if (sortField !== field) return null;
    return sortDir === 'asc' ? ' 🔼' : ' 🔽';
  };

  return (
    <div className="replays-table-container glass-card">
      <div className="table-header-controls">
        <div className="table-title">
          <h3>Replays in Dataset ({replays.length})</h3>
        </div>
        <div className="table-search">
          <input
            type="text"
            className="filter-input"
            placeholder="Filter map title, artist, creator..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
        </div>
      </div>

      <div className="table-scroll-wrapper">
        <table className="replays-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('title')}>Beatmap{renderSortIndicator('title')}</th>
              <th onClick={() => handleSort('star_rating')}>SR{renderSortIndicator('star_rating')}</th>
              <th>Mods</th>
              <th onClick={() => handleSort('accuracy')}>Acc{renderSortIndicator('accuracy')}</th>
              <th onClick={() => handleSort('misses')}>Misses{renderSortIndicator('misses')}</th>
              <th onClick={() => handleSort('max_combo')}>Combo{renderSortIndicator('max_combo')}</th>
              <th onClick={() => handleSort('mastery_score')}>Mastery Score{renderSortIndicator('mastery_score')}</th>
            </tr>
          </thead>
          <tbody>
            {filteredAndSortedReplays.length === 0 ? (
              <tr>
                <td colSpan="7" className="empty-table-msg">
                  No replays found matching filters.
                </td>
              </tr>
            ) : (
              filteredAndSortedReplays.map((replay, idx) => {
                const sr = (replay.star_rating || 0).toFixed(2);
                const acc = ((replay.accuracy || 0) * 100).toFixed(2);
                const modsStr = parseMods(replay.mods);
                const mastery = (replay.mastery_score || 0).toFixed(4);

                const mapUrl = replay.beatmap_id
                  ? `https://osu.ppy.sh/b/${replay.beatmap_id}`
                  : replay.mapset_id
                  ? `https://osu.ppy.sh/beatmapsets/${replay.mapset_id}`
                  : `https://osu.ppy.sh/beatmapsets?q=${encodeURIComponent(replay.title || '')}`;

                return (
                  <tr key={replay.replay_hash || idx} className="replay-row">
                    <td className="map-info-cell">
                      <div className="map-title">
                        <a
                          href={mapUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="map-title-link"
                        >
                          {replay.title}
                        </a>
                      </div>
                      <div className="map-sub">
                        <span className="artist">{replay.artist}</span>
                        <span className="diff-name">[{replay.difficulty}]</span>
                      </div>
                    </td>
                    <td>
                      <span className="sr-badge" style={{ backgroundColor: getSRColor(replay.star_rating) }}>
                        ★ {sr}
                      </span>
                    </td>
                    <td>
                      <span className={`mod-badge mod-${modsStr}`}>{modsStr}</span>
                    </td>
                    <td className="stat-acc">{acc}%</td>
                    <td className={`stat-misses ${replay.misses > 0 ? 'has-misses' : 'fc'}`}>
                      {replay.misses}
                    </td>
                    <td className="stat-combo">{replay.max_combo}x</td>
                    <td className="stat-mastery">
                      <span className="mastery-val">{mastery}</span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Helper to color star rating badges
function getSRColor(sr) {
  if (!sr) return '#4A5568';
  if (sr < 2.0) return '#48BB78'; // Green
  if (sr < 3.5) return '#4299E1'; // Blue
  if (sr < 5.0) return '#ECC94B'; // Yellow
  if (sr < 6.5) return '#ED8936'; // Orange
  if (sr < 8.0) return '#F56565'; // Red
  return '#9F7AEA'; // Purple / Expert
}

export default ReplaysTable;
