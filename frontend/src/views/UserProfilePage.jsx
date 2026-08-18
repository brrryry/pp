import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fetchUser, fetchUserReplays, fetchUserRecommendations, fetchJobStatus, recalibrateUser } from '../services/api';
import ReplaysTable from '../components/ReplaysTable';
import FogOfWarGraph from '../components/FogOfWarGraph';
import UserSearch from '../components/UserSearch';

function UserProfilePage() {
  const { osu_id } = useParams();

  const [userInfo, setUserInfo] = useState(null);
  const [replays, setReplays] = useState([]);
  const [recommendedMaps, setRecommendedMaps] = useState([]);

  const [loading, setLoading] = useState(true);
  const [recsLoading, setRecsLoading] = useState(false);
  const [recalibrating, setRecalibrating] = useState(false);
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobInfo, setJobInfo] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [activeTab, setActiveTab] = useState('fog'); // 'fog' | 'replays'

  const handleRecalibrate = async () => {
    if (!osu_id || recalibrating) return;
    setRecalibrating(true);
    setErrorMsg(null);
    try {
      const res = await recalibrateUser(osu_id);
      if (res.job_id) {
        setActiveJobId(res.job_id);
      }
    } catch (err) {
      console.error('Recalibration failed:', err);
      setErrorMsg(err.message || 'Failed to start recalibration');
    } finally {
      setRecalibrating(false);
    }
  };

  // Fetch recommendations for current user
  const loadRecommendations = useCallback(async (targetId) => {
    setRecsLoading(true);
    try {
      const data = await fetchUserRecommendations(targetId, 10);
      if (data && data.recommended_maps) {
        setRecommendedMaps(data.recommended_maps);
      }
    } catch (err) {
      console.error('Failed to load recommendations:', err);
    } finally {
      setRecsLoading(false);
    }
  }, []);

  // Main data loader function
  const loadProfile = useCallback(async () => {
    if (!osu_id) return;
    setLoading(true);
    setErrorMsg(null);
    setActiveJobId(null);
    setJobInfo(null);

    try {
      // 1. Fetch User
      const uRes = await fetchUser(osu_id);
      if (uRes.job_id) {
        setActiveJobId(uRes.job_id);
        setLoading(false);
        return;
      }
      if (uRes.user) {
        setUserInfo(uRes.user);
      }

      // 2. Fetch Replays
      const rRes = await fetchUserReplays(osu_id);
      if (rRes.job_id) {
        setActiveJobId(rRes.job_id);
        setLoading(false);
        return;
      }

      if (rRes.replays) {
        setReplays(rRes.replays);
      }

      // 3. Load Fog of War Recommendations
      await loadRecommendations(osu_id);

    } catch (err) {
      console.error('Error loading user profile:', err);
      setErrorMsg(err.message || 'Failed to load user profile');
    } finally {
      setLoading(false);
    }
  }, [osu_id, loadRecommendations]);

  // Initial load when osu_id changes
  useEffect(() => {
    loadProfile();
  }, [osu_id, loadProfile]);

  // Polling effect for ingestion job
  useEffect(() => {
    if (!activeJobId) return;

    let isSubscribed = true;

    const checkStatus = async () => {
      try {
        const statusData = await fetchJobStatus(activeJobId);
        if (!isSubscribed) return;
        setJobInfo(statusData);

        if (statusData.status === 'ready') {
          setActiveJobId(null);
          loadProfile(); // Reload full profile now that job is done
        } else if (statusData.status === 'failed') {
          setActiveJobId(null);
          setErrorMsg(statusData.error || 'Replay ingestion job failed');
        }
      } catch (err) {
        console.error('Job status polling error:', err);
      }
    };

    // Immediate initial check
    checkStatus();

    // Poll once every 30 seconds (30,000 ms)
    const interval = setInterval(checkStatus, 30000);

    return () => {
      isSubscribed = false;
      clearInterval(interval);
    };
  }, [activeJobId, loadProfile]);

  return (
    <div className="user-profile-page container">
      {/* Top Navigation / Header */}
      <header className="profile-header glass-card">
        <div className="header-left">
          <Link to="/users/" className="back-link">
            ← Back to Search
          </Link>
          <div className="user-avatar-meta">
            <img
              src={`https://a.ppy.sh/${userInfo?.osu_id || osu_id}`}
              alt="User Avatar"
              className="user-avatar"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = 'https://osu.ppy.sh/images/layout/avatar-placeholder.png';
              }}
            />
            <div className="user-text">
              <h2 className="username">{userInfo?.username || `User #${osu_id}`}</h2>
              <span className="user-id">osu! ID: {userInfo?.osu_id || osu_id}</span>
            </div>
          </div>
        </div>

        <div className="header-right">
          <div className="quick-search-wrapper">
            <UserSearch initialValue={osu_id} />
          </div>
        </div>
      </header>

      {/* Main Display Area */}
      {loading ? (
        <div className="loading-state glass-card text-center">
          <div className="spinner-large">🔄</div>
          <h3>Loading Profile #{osu_id}...</h3>
          <p>Fetching user information and dataset replays...</p>
        </div>
      ) : activeJobId ? (
        <div className="job-polling-state glass-card text-center">
          <div className="pulse-spinner">⚙️</div>
          <h2>Ingesting Top & Recent Plays for User #{osu_id}</h2>
          <p className="job-sub">
            Retrieving top 100 best plays and 50 recent plays via osu! API...
          </p>
          <div className="job-status-badge">
            Status: <strong>{jobInfo?.status || 'queued'}</strong>
            {jobInfo?.processed && ` • Processed ${jobInfo.processed} plays`}
          </div>
          <p className="job-tip">Please wait a moment while the dataset updates automatically.</p>
        </div>
      ) : errorMsg ? (
        <div className="error-state glass-card text-center">
          <span className="error-icon">⚠️</span>
          <h2>Unable to Load Profile</h2>
          <p className="error-detail">{errorMsg}</p>
          <button className="btn-retry" onClick={loadProfile}>
            🔄 Retry
          </button>
        </div>
      ) : (
        <div className="profile-content">
          {/* Overview Navigation Tabs */}
          <div className="view-tabs">
            <button
              className={`tab-btn ${activeTab === 'fog' ? 'active' : ''}`}
              onClick={() => setActiveTab('fog')}
            >
              🌫️ Recommendation Map
            </button>
            <button
              className={`tab-btn ${activeTab === 'replays' ? 'active' : ''}`}
              onClick={() => setActiveTab('replays')}
            >
              📊 Replays Dataset ({replays.length})
            </button>
            <button
              className="tab-btn btn-recalibrate"
              onClick={handleRecalibrate}
              disabled={recalibrating || !!activeJobId}
              title="Re-fetch top 100 best plays and 50 recent plays from osu! API"
              style={{ marginLeft: 'auto' }}
            >
              {recalibrating ? '🔄 Starting...' : '⚡ Recalibrate Plays'}
            </button>
          </div>

          {/* Tab Views */}
          {activeTab === 'fog' && (
            <div className="tab-pane">
              <FogOfWarGraph
                recommendedMaps={recommendedMaps}
                isLoading={recsLoading}
                username={userInfo?.username || `User #${osu_id}`}
              />
            </div>
          )}

          {activeTab === 'replays' && (
            <div className="tab-pane">
              <ReplaysTable replays={replays} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default UserProfilePage;
