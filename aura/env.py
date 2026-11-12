from __future__ import annotations

import math

import numpy as np

BITRATES = np.array([0.30, 0.75, 1.20, 1.85, 2.85, 4.30, 8.0, 16.0, 25.0, 40.0])
LOG_BITRATES = np.log(BITRATES)
CHUNK_SECONDS = 4.0
CHUNKS_PER_SESSION = 50
MAX_BUFFER = 30.0
INITIAL_BUFFER = 4.0

REBUFFER_WEIGHT = 4.3
SWITCH_WEIGHT = 1.0
LATENCY_WEIGHT = 0.5

EDGE_CAPACITY = np.array([8.0, 16.0, 40.0])
EDGE_DELAY_MS = np.array([5.0, 15.0, 35.0])
EDGE_LOAD_MEAN = np.array([0.55, 0.40, 0.25])

HISTORY = 5
LOG_SCALE = math.log1p(100.0)
MIN_RATE = 0.01


class EdgeLoadProcess:

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.load = np.clip(EDGE_LOAD_MEAN + rng.normal(0, 0.08, 3), 0.02, 0.97)

    def step(self):
        noise = self.rng.normal(0, 0.05, 3)
        spikes = (self.rng.random(3) < 0.03) * self.rng.uniform(0.2, 0.4, 3)
        self.load = np.clip(self.load + 0.15 * (EDGE_LOAD_MEAN - self.load) + noise + spikes,
                            0.02, 0.97)


class JointEnv:

    n_edges = 3
    n_bitrates = len(BITRATES)
    n_actions = n_bitrates * n_edges

    def __init__(self, window, context: bool = True, migration_delay: float = 1.0,
                 seed: int = 0, start: float = 0.0, planner=None, planner_dim: int = 0):
        self.w = window
        self.context = context
        self.migration_delay = migration_delay
        self.seed = seed
        self.start = start
        self.planner = planner
        self.obs_dim = (20 if context else 17) + (planner_dim if planner is not None else 0)
        self.plan = None
        self._ends = np.append(window.times[1:], window.duration)
        self.reset()


    def _index(self, clock):
        return int(np.searchsorted(self.w.times, clock % self.w.duration, side="right") - 1)

    def _transfer(self, clock, megabits, edge_cap):
        remaining, elapsed, radio_area = megabits, 0.0, 0.0
        t = clock
        for _ in range(100000):
            local = t % self.w.duration
            i = int(np.searchsorted(self.w.times, local, side="right") - 1)
            seg = max(self._ends[i] - local, 1e-6)
            radio = max(self.w.throughput[i], MIN_RATE)
            rate = max(min(radio, edge_cap), MIN_RATE)
            if rate * seg >= remaining:
                dt = remaining / rate
                return elapsed + dt, (radio_area + radio * dt) / (elapsed + dt)
            remaining -= rate * seg
            elapsed += seg
            radio_area += radio * seg
            t += seg
        return elapsed, radio_area / max(elapsed, 1e-6)


    def edge_available(self):
        return EDGE_CAPACITY * (1.0 - self.loads.load)

    def edge_latency(self):
        load = self.loads.load
        queue = np.minimum(EDGE_DELAY_MS * load / (1.0 - load), 200.0)
        return EDGE_DELAY_MS + queue

    def radio_latency(self):
        return float(self.w.latency[self._index(self.clock)])

    def handover_risk(self):
        return float(self.w.handover_risk[self._index(self.clock)])

    def radio_estimate(self):
        hist = np.asarray(self.cap_history[-HISTORY:])
        return float(len(hist) / np.sum(1.0 / np.maximum(hist, MIN_RATE)))

    def estimate_error(self):
        return float(max(self.errors[-HISTORY:])) if self.errors else 0.0


    def reset(self):
        self.t = 0
        self.clock = self.start
        self.buffer = INITIAL_BUFFER
        self.prev_bitrate = None
        self.prev_edge = None
        self.loads = EdgeLoadProcess(np.random.default_rng(self.seed))
        self.cap_history = [float(self.w.throughput[self._index(self.clock)])]
        self.last_latency = self.radio_latency()
        self.errors = []
        return self.observation()

    def observation(self):
        est = self.radio_estimate()
        prev_idx = 0 if self.prev_bitrate is None else int(np.argmin(np.abs(BITRATES - self.prev_bitrate)))
        edge_onehot = np.zeros(3)
        if self.prev_edge is not None:
            edge_onehot[self.prev_edge] = 1.0
        obs = [
            math.log1p(est) / LOG_SCALE,
            math.log1p(self.cap_history[-1]) / LOG_SCALE,
            self.last_latency / 200.0,
            self.buffer / MAX_BUFFER,
            prev_idx / (self.n_bitrates - 1),
            *edge_onehot,
            *(np.log1p(self.edge_available()) / LOG_SCALE),
            *(np.log1p(np.minimum(est, self.edge_available())) / LOG_SCALE),
            *(self.edge_latency() / 100.0),
        ]
        if self.context:
            i = self._index(self.clock)
            obs += [self.w.speed[i] / 120.0,
                    (self.w.rsrp[i] + 140.0) / 90.0,
                    self.w.handover_risk[i]]
        if self.planner is not None:
            features, self.plan = self.planner(self)
            obs += list(features)
        return np.asarray(obs, dtype=np.float32)

    def shifted(self, action, delta):
        idx, edge = self.decode(action)
        return self.encode(int(np.clip(idx + delta, 0, self.n_bitrates - 1)), edge)

    def decode(self, action):
        return int(action) % self.n_bitrates, int(action) // self.n_bitrates

    @staticmethod
    def encode(bitrate_idx, edge):
        return int(edge) * len(BITRATES) + int(bitrate_idx)

    def simulate(self, bitrate_idx, edge):
        bitrate = BITRATES[bitrate_idx]
        latency = self.radio_latency() + self.edge_latency()[edge]
        migrated = self.prev_edge is not None and edge != self.prev_edge
        idle = latency / 1000.0 + (self.migration_delay if migrated else 0.0)
        transfer, radio_mean = self._transfer(self.clock + idle, bitrate * CHUNK_SECONDS,
                                              self.edge_available()[edge])
        download = idle + transfer
        rebuffer = max(download - self.buffer, 0.0)
        switch = 0.0 if self.prev_bitrate is None else abs(math.log(bitrate) - math.log(self.prev_bitrate))
        reward = (math.log(bitrate) - REBUFFER_WEIGHT * rebuffer
                  - SWITCH_WEIGHT * switch - LATENCY_WEIGHT * latency / 100.0)
        return reward, download, rebuffer, latency, radio_mean, migrated

    def step(self, action):
        bitrate_idx, edge = self.decode(action)
        estimate = self.radio_estimate()
        reward, download, rebuffer, latency, radio_mean, migrated = self.simulate(bitrate_idx, edge)

        self.clock += download
        self.buffer = max(self.buffer - download, 0.0) + CHUNK_SECONDS
        if self.buffer > MAX_BUFFER:
            self.clock += self.buffer - MAX_BUFFER
            self.buffer = MAX_BUFFER

        info = {
            "bitrate": float(BITRATES[bitrate_idx]),
            "edge": edge,
            "rebuffer": rebuffer,
            "latency": latency,
            "migrated": migrated,
            "bitrate_switch": self.prev_bitrate is not None and BITRATES[bitrate_idx] != self.prev_bitrate,
        }
        self.errors.append(abs(estimate - radio_mean) / max(radio_mean, MIN_RATE))
        self.cap_history.append(radio_mean)
        self.last_latency = latency - self.edge_latency()[edge]
        self.prev_bitrate = float(BITRATES[bitrate_idx])
        self.prev_edge = edge
        self.loads.step()
        self.t += 1
        done = self.t >= CHUNKS_PER_SESSION
        obs = np.zeros(self.obs_dim, dtype=np.float32) if done else self.observation()
        return obs, reward, done, info
