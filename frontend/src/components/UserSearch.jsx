import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function UserSearch({ initialValue = '' }) {
  const [inputVal, setInputVal] = useState(initialValue);
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = inputVal.trim();
    if (trimmed) {
      navigate(`/users/${encodeURIComponent(trimmed)}`);
    }
  };

  const handleQuickSelect = (id) => {
    setInputVal(id);
    navigate(`/users/${id}`);
  };

  return (
    <div className="search-box-container">
      <form onSubmit={handleSubmit} className="search-form">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Enter osu! User ID or Username..."
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
          />
          <button type="submit" className="search-button">
            Analyze User
          </button>
        </div>
      </form>
      <div className="quick-select-pills">
        <span className="quick-label">Try popular profiles:</span>
        <button className="pill-btn" onClick={() => handleQuickSelect('2')}>peppy (2)</button>
        <button className="pill-btn" onClick={() => handleQuickSelect('7562902')}>mrekk (7562902)</button>
        <button className="pill-btn" onClick={() => handleQuickSelect('3837900')}>WhiteCat (3837900)</button>
      </div>
    </div>
  );
}

export default UserSearch;
