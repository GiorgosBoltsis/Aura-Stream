from __future__ import annotations

import numpy as np

from .env import CHUNK_SECONDS, JointEnv

EVAL_SEED = 12345


def run_episode(policy, window, context=True, migration_delay=1.0, seed=0, planner=None,
                planner_dim=0):
    env = JointEnv(window, context=context, migration_delay=migration_delay, seed=seed,
                   planner=planner, planner_dim=planner_dim)
    obs = env.reset()
    rewards, bitrates, rebuffer, latency, migrations, switches = [], [], 0.0, [], 0, 0
    done = False
    while not done:
        action = policy(env, obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        bitrates.append(info["bitrate"])
        rebuffer += info["rebuffer"]
        latency.append(info["latency"])
        migrations += int(info["migrated"])
        switches += int(info["bitrate_switch"])
    n = len(rewards)
    return {
        "qoe": float(np.mean(rewards)),
        "bitrate": float(np.mean(bitrates)),
        "rebuffer_ratio": rebuffer / (n * CHUNK_SECONDS),
        "latency_ms": float(np.mean(latency)),
        "migrations_per_100": 100.0 * migrations / n,
        "bitrate_switches_per_100": 100.0 * switches / n,
    }


def evaluate(policy, windows, context=True, migration_delay=1.0, planner=None, planner_dim=0):
    rows = []
    for i, w in enumerate(windows):
        m = run_episode(policy, w, context, migration_delay, seed=EVAL_SEED + i,
                        planner=planner, planner_dim=planner_dim)
        m.update(dataset=w.dataset, window=i)
        rows.append(m)
    return rows


def paired_bootstrap(a, b, n_resamples=10000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(p, 1.0)), "win_rate": float((diff > 0).mean())}
