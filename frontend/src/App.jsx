import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import UserSearchPage from './views/UserSearchPage';
import UserProfilePage from './views/UserProfilePage';

function App() {
  return (
    <BrowserRouter>
      <div className="glass-bg"></div>
      <Routes>
        <Route path="/" element={<UserSearchPage />} />
        <Route path="/users/" element={<UserSearchPage />} />
        <Route path="/users/:osu_id" element={<UserProfilePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
