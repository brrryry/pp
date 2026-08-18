#!/usr/bin/env python3
"""
Flush Redis Job Queues, Active Job Locks, and Caches script for Osu! PP Profiler (Dev or Prod).
Usage:
    python server/flush_cache_prod.py [--env dev|prod] [--redis-url REDIS_URL] [--host HOST] [--port PORT]
"""

import os
import sys
import argparse

try:
    import redis
except ImportError:
    print("Error: redis is required. Run 'pip install redis'")
    sys.exit(1)

try:
    from rq import Queue
    HAS_RQ = True
except ImportError:
    HAS_RQ = False

def get_redis_urls_for_env(env_target="prod", custom_url=None, host=None, port=None):
    urls = []
    if custom_url:
        urls.append(custom_url)
    
    # Environment variables
    if os.environ.get("REDIS_URL"):
        urls.append(os.environ.get("REDIS_URL"))

    # Read server/.env
    env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", ".env"),
        os.path.join(os.getcwd(), "server", ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("REDIS_URL="):
                        val = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            urls.append(val)

    if env_target.lower() == "dev":
        target_host = host or "localhost"
        target_port = port or 6380
        urls.extend([
            f"redis://{target_host}:{target_port}/0",
            "redis://localhost:6380/0",
            "redis://127.0.0.1:6380/0",
            "redis://redis:6380/0"
        ])
    else:
        target_host = host or "bryan-nas.gkhomenetwork.lan"
        target_port = port or 6380
        urls.extend([
            f"redis://{target_host}:{target_port}/0",
            f"redis://{target_host}:6379/0",
            "redis://localhost:6380/0",
            "redis://localhost:6379/0",
            "redis://redis:6379/0"
        ])

    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)

    return deduped

def flush_redis_cache(env_target="prod", redis_url=None, host=None, port=None):
    candidate_urls = get_redis_urls_for_env(env_target, redis_url, host, port)

    r = None
    connected_url = None
    for url in candidate_urls:
        try:
            client = redis.from_url(url, socket_connect_timeout=3)
            client.ping()
            r = client
            connected_url = url
            break
        except Exception:
            continue

    if not r:
        print(f"[ERROR] Could not connect to Redis server ({env_target.upper()}) at any of: {candidate_urls}")
        sys.exit(1)

    print(f"[INFO] Connected to {env_target.upper()} Redis at: {connected_url}")

    # 1. Clear RQ Queues
    if HAS_RQ:
        for queue_name in ['top_replay_ingestion', 'beatmap_ingestion', 'replay_ingestion', 'failed']:
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
        'cache:user:*',
        'cache:user_id:*',
        'cache:map_failed:*',
        'beatmap:hash:*',
        'beatmap:*',
        'ratelimit:*'
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

    print(f"\n[SUCCESS] Redis ({env_target.upper()}) job queues, active locks, and all recommendation caches flushed! (Total keys removed: {deleted_count})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flush Redis queues, locks, and caches (Dev or Prod)")
    parser.add_argument("--env", "--target", choices=["dev", "prod"], default="prod", help="Target environment: 'dev' (localhost:6380) or 'prod' (bryan-nas.gkhomenetwork.lan:6380) (default: prod)")
    parser.add_argument("--redis-url", default=None, help="Custom Redis URL (overrides --env setting)")
    parser.add_argument("--host", default=None, help="Custom Redis host")
    parser.add_argument("--port", type=int, default=None, help="Custom Redis port")
    args = parser.parse_args()

    flush_redis_cache(env_target=args.env, redis_url=args.redis_url, host=args.host, port=args.port)
