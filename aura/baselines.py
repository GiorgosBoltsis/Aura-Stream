from __future__ import annotations

import itertools
import math

import numpy as np

from .env import (BITRATES, CHUNK_SECONDS, LATENCY_WEIGHT, LOG_BITRATES, MAX_BUFFER,
                  REBUFFER_WEIGHT, SWITCH_WEIGHT, JointEnv)

SAFETY = 0.9


def _highest_below(rate):
    idx = np.nonzero(BITRATES <= rate)[0]
    return int(idx[-1]) if len(idx) else 0


def _context_edge(env: JointEnv, margin: float = 0.15):
    est = env.radio_estimate()
    score = (np.log(np.minimum(est, env.edge_available()))
             - LATENCY_WEIGHT * (env.radio_latency() + env.edge_latency()) / 100.0)
    best = int(np.argmax(score))
    if env.prev_edge is not None and best != env.prev_edge:
        if score[best] - score[env.prev_edge] < margin + env.migration_delay * 0.2:
            return env.prev_edge
    return best


def rate_nearest(env: JointEnv, obs=None):
    rate = SAFETY * min(env.radio_estimate(), env.edge_available()[0])
    return env.encode(_highest_below(rate), 0)


def rate_least_loaded(env: JointEnv, obs=None):
    edge = int(np.argmin(env.loads.load))
    rate = SAFETY * min(env.radio_estimate(), env.edge_available()[edge])
    return env.encode(_highest_below(rate), edge)


def rate_context_edge(env: JointEnv, obs=None):
    edge = _context_edge(env)
    rate = SAFETY * min(env.radio_estimate(), env.edge_available()[edge])
    return env.encode(_highest_below(rate), edge)


def bba_context_edge(env: JointEnv, obs=None, reservoir=5.0, cushion=15.0):
    edge = _context_edge(env)
    cap_idx = _highest_below(1.5 * min(env.radio_estimate(), env.edge_available()[edge]))
    frac = np.clip((env.buffer - reservoir) / cushion, 0.0, 1.0)
    idx = int(round(frac * cap_idx))
    return env.encode(idx, edge)


def context_joint(env: JointEnv, obs=None):
    edge = _context_edge(env)
    rate = SAFETY * min(env.radio_estimate(), env.edge_available()[edge])
    if env.handover_risk() > 0.5:
        rate *= 0.8
    if env.buffer < 8.0:
        rate *= 0.7
    return env.encode(_highest_below(rate), edge)


_SEQUENCES = np.array(list(itertools.product(range(len(BITRATES)), repeat=3)))


def mpc_plan(env: JointEnv, horizon_sequences=_SEQUENCES):
    est = env.radio_estimate() / (1.0 + min(env.estimate_error(), 10.0))
    available = env.edge_available()
    latency = env.radio_latency() + env.edge_latency()
    rates = BITRATES[horizon_sequences]
    logs = LOG_BITRATES[horizon_sequences]
    prev_log = None if env.prev_bitrate is None else math.log(env.prev_bitrate)
    first = np.zeros(env.n_edges, dtype=int)
    values = np.zeros(env.n_edges)
    for edge in range(env.n_edges):
        thr = max(min(est, available[edge]), 1e-3)
        migrate = env.prev_edge is not None and edge != env.prev_edge
        buffer = np.full(len(horizon_sequences), env.buffer)
        value = np.zeros(len(horizon_sequences))
        for step in range(horizon_sequences.shape[1]):
            download = rates[:, step] * CHUNK_SECONDS / thr + latency[edge] / 1000.0
            if step == 0 and migrate:
                download = download + env.migration_delay
            rebuffer = np.maximum(download - buffer, 0.0)
            buffer = np.minimum(np.maximum(buffer - download, 0.0) + CHUNK_SECONDS, MAX_BUFFER)
            if step == 0:
                switch = 0.0 if prev_log is None else np.abs(logs[:, 0] - prev_log)
            else:
                switch = np.abs(logs[:, step] - logs[:, step - 1])
            value += (logs[:, step] - REBUFFER_WEIGHT * rebuffer - SWITCH_WEIGHT * switch
                      - LATENCY_WEIGHT * latency[edge] / 100.0)
        i = int(np.argmax(value))
        first[edge], values[edge] = horizon_sequences[i, 0], value[i]
    return first, values


def mpc_joint(env: JointEnv, obs=None):
    first, values = mpc_plan(env)
    edge = int(np.argmax(values))
    return env.encode(first[edge], edge)


def oracle(env: JointEnv, obs=None):
    best_reward, best_action = -np.inf, 0
    for edge in range(env.n_edges):
        for b in range(len(BITRATES)):
            reward = env.simulate(b, edge)[0]
            if reward > best_reward:
                best_reward, best_action = reward, env.encode(b, edge)
    return best_action


HEURISTICS = {
    "Rate+Nearest": rate_nearest,
    "Rate+LeastLoaded": rate_least_loaded,
    "Rate+CtxEdge": rate_context_edge,
    "BBA+CtxEdge": bba_context_edge,
    "Ctx-Joint": context_joint,
    "MPC-Joint": mpc_joint,
}
