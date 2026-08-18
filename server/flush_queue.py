#!/usr/bin/env python3
"""
Flush Redis Job Queues, Active Job Locks, and Caches script for Osu! PP Profiler.
Run this script anytime to completely clear queued worker jobs, active job locks, and endpoint caches.

Usage:
    python server/flush_queue.py
    OR
    python flush_queue.py
"""

import os
import sys
# pyrefly: ignore [missing-import]
import redis
# pyrefly: ignore [missing-import]
from rq import Queue

def flush_redis():
    # Comprehensive connection URLs covering host localhost, docker network redis, ports 6379 & 6380
    redis_urls = [
        os.environ.get("REDIS_URL"),
        "redis://redis:6379/0",
        "redis://localhost:6380/0",
        "redis://localhost:6379/0",
        "redis://redis:6380/0"
    ]
    # Filter out Nones
    redis_urls = [u for u in redis_urls if u]

    r = None
    connected_url = None
    for url in redis_urls:
        try:
            client = redis.from_url(url, socket_connect_timeout=2)
            client.ping()
            r = client
            connected_url = url
            break
        except Exception:
            continue

    if not r:
        print("[ERROR] Could not connect to Redis server.")
        sys.exit(1)

    print(f"[INFO] Connected to Redis at: {connected_url}")

    # 1. Clear RQ Queues
    for queue_name in ['top_replay_ingestion', 'beatmap_ingestion', 'failed']:
        try:
            q = Queue(queue_name, connection=r)
            q.empty()
            print(f"[OK] Cleared RQ Queue: '{queue_name}'")
        except Exception as e:
            print(f"[WARN] Could not clear queue '{queue_name}': {e}")

    # 2. Delete all job status, locks, and cache key patterns
    patterns = [
        'rq:job:*',
        'rq:queue:*',
        'job:*',
        'active_job:*',
        'lock:*',
        'cache:*',
        'cache:recs:*',
        'cache:endpoint:recs:*',
        'cache:map:*',
        'cache:user_replays:*',
        'cache:comfort_sr:*',
        'beatmap:hash:*',
        'beatmap:*'
    ]

    deleted_count = 0
    for pattern in patterns:
        try:
            keys = list(r.scan_iter(pattern))
            if keys:
                r.delete(*keys)
                deleted_count += len(keys)
                print(f"[OK] Deleted {len(keys)} keys matching '{pattern}'")
        except Exception as e:
            print(f"[WARN] Failed to delete pattern '{pattern}': {e}")

    # 3. Complete FLUSHALL to guarantee total cache reset
    try:
        r.flushall()
        print("[OK] Executed FLUSHALL: All Redis databases and cached recommendations wiped cleanly!")
    except Exception as e:
        print(f"[WARN] FLUSHALL warning: {e}")

    print(f"\n[SUCCESS] Redis job queues, active locks, and all recommendation caches flushed! (Total keys removed: {deleted_count})")


if __name__ == '__main__':
    flush_redis()


