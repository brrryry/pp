// API service helper for osu! PP & Skill Profiler

const BASE_URL = ''; // Relative paths rely on Vite proxy in dev, same origin in prod

export async function fetchUser(osuId) {
  const res = await fetch(`${BASE_URL}/users?osu_id=${encodeURIComponent(osuId)}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch user (${res.status})`);
  }
  return await res.json();
}

export async function fetchUserReplays(osuId) {
  const res = await fetch(`${BASE_URL}/user/replays?osu_id=${encodeURIComponent(osuId)}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch user replays (${res.status})`);
  }
  return await res.json();
}

export async function fetchUserRecommendations(osuId, k = 10) {
  const res = await fetch(`${BASE_URL}/user/recommended_maps?osu_id=${encodeURIComponent(osuId)}&k=${k}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch recommendations (${res.status})`);
  }
  return await res.json();
}

export async function fetchJobStatus(jobId) {
  const res = await fetch(`${BASE_URL}/jobs?job_id=${encodeURIComponent(jobId)}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch job status (${res.status})`);
  }
  return await res.json();
}

export async function recalibrateUser(osuId) {
  const res = await fetch(`${BASE_URL}/user/recalibrate?osu_id=${encodeURIComponent(osuId)}`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error(`Failed to trigger recalibration (${res.status})`);
  }
  return await res.json();
}
