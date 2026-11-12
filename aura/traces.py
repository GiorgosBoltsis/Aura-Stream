from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEGMENT_SECONDS = 300.0

CONFIWENT_DIRS = {
    "hsdpa": ["traces_3gp"],
    "fcc": ["fcc_ori/cooked_traces"],
    "oboe": ["traces_oboe"],
    "puffer": ["puffer_211017/cooked_traces", "puffer_220218/cooked_traces"],
}
DATASETS = ["hsdpa", "fcc", "oboe", "puffer", "5g"]
DEFAULT_LATENCY_MS = {"5G": 25.0, "LTE": 40.0, "HSPA+": 60.0}


@dataclass
class Window:

    dataset: str
    trace_id: str
    times: np.ndarray
    duration: float
    throughput: np.ndarray
    latency: np.ndarray
    speed: np.ndarray
    rsrp: np.ndarray
    handover_risk: np.ndarray
    real_context: bool


def _seed_from(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def _handover_risk(handover_events: np.ndarray, horizon: int = 30) -> np.ndarray:
    counts = np.convolve(handover_events.astype(float), np.ones(horizon))[: len(handover_events)]
    return np.clip(counts / 3.0, 0.0, 1.0)


def _synthetic_context(n: int, seed: int):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0, 100)
    speed = np.empty(n)
    s = target
    for i in range(n):
        if rng.random() < 0.02:
            target = rng.uniform(0, 100)
        s = s + 0.1 * (target - s) + rng.normal(0, 2)
        speed[i] = s
    speed = np.clip(speed, 0, 130)
    rsrp = np.clip(-85 - 0.12 * speed + np.cumsum(rng.normal(0, 0.6, n)) * 0.3
                   + rng.normal(0, 3, n), -125, -60)
    handovers = rng.random(n) < (0.002 + speed / 130 * 0.03)
    latency = np.clip(30 + 0.08 * speed + rng.normal(0, 5, n), 8, 200)
    return latency, speed, rsrp, _handover_risk(handovers)


def _load_confiwent_trace(path: str):
    try:
        arr = np.loadtxt(path)
    except Exception:
        return None
    if arr.ndim != 2 or len(arr) < 20:
        return None
    times = arr[:, 0].astype(float) - arr[0, 0]
    keep = np.concatenate([[True], np.diff(times) > 0])
    return times[keep], np.clip(arr[keep, 1].astype(float), 0.01, 200.0)


def _load_5g_trace(path: str):
    df = pd.read_csv(path)
    thr = pd.to_numeric(df["DL_bitrate"], errors="coerce").ffill().fillna(0) / 1000.0
    speed = pd.to_numeric(df["Speed"], errors="coerce").ffill().fillna(0)
    rsrp = pd.to_numeric(df["RSRP"], errors="coerce").ffill().bfill().fillna(-100)
    cell = df["CellID"].astype(str)
    handovers = (cell != cell.shift()).to_numpy().copy()
    handovers[0] = False

    ping = pd.to_numeric(df["PINGAVG"], errors="coerce")
    mode_default = df["NetworkMode"].map(DEFAULT_LATENCY_MS).fillna(50.0)
    latency = ping.interpolate(limit_direction="both")
    latency = latency.fillna(mode_default).clip(8, 300)

    times = np.arange(len(df), dtype=float)
    return (times, np.clip(thr.to_numpy(), 0.01, 200.0), latency.to_numpy(),
            np.clip(speed.to_numpy(), 0, 130), np.clip(rsrp.to_numpy(), -140, -50),
            _handover_risk(handovers))


def _slice(dataset, trace_id, times, arrays, real_context):
    step = np.median(np.diff(times)) if len(times) > 1 else 1.0
    total = times[-1] + step
    if total <= 1.5 * SEGMENT_SECONDS:
        bounds = [(0.0, total)]
    else:
        n = int(total // SEGMENT_SECONDS)
        bounds = [(i * SEGMENT_SECONDS, (i + 1) * SEGMENT_SECONDS) for i in range(n)]
    out = []
    for lo, hi in bounds:
        mask = (times >= lo) & (times < hi)
        if mask.sum() < 10:
            continue
        seg_times = times[mask] - lo
        seg = [a[mask] for a in arrays]
        out.append(Window(dataset, f"{trace_id}@{int(lo)}", seg_times, hi - lo, *seg, real_context))
    return out


def _dataset_files(dataset: str, root: str):
    if dataset == "5g":
        base = os.path.join(root, "ucc5g", "5G-production-dataset", "Download")
        groups = {m: sorted(glob.glob(os.path.join(base, m, "*.csv")))
                  for m in ("Driving", "Static")}
        return groups
    files = []
    for sub in CONFIWENT_DIRS[dataset]:
        files += sorted(f for f in glob.glob(os.path.join(root, "confiwent_repo", sub, "*"))
                        if os.path.isfile(f))
    return {"all": files}


def load_dataset(dataset: str, root: str, train_frac: float = 0.7,
                 max_train: int = 400, max_test: int = 120, split_seed: int = 0):
    rng = np.random.default_rng(split_seed + _seed_from(dataset) % 1000)
    train, test = [], []
    for group, files in _dataset_files(dataset, root).items():
        files = list(files)
        rng.shuffle(files)
        n_train = max(1, int(round(train_frac * len(files))))
        for i, path in enumerate(files):
            trace_id = os.path.relpath(path, root)
            if dataset == "5g":
                times, *arrays = _load_5g_trace(path)
                real = True
            else:
                loaded = _load_confiwent_trace(path)
                if loaded is None:
                    continue
                times, thr = loaded
                arrays = (thr, *_synthetic_context(len(thr), _seed_from(trace_id)))
                real = False
            windows = _slice(dataset, trace_id, times, arrays, real)
            (train if i < n_train else test).extend(windows)

    def cap(ws, k, salt):
        if len(ws) <= k:
            return ws
        idx = np.random.default_rng(salt).choice(len(ws), k, replace=False)
        return [ws[j] for j in sorted(idx)]

    return cap(train, max_train, 1), cap(test, max_test, 2)


def load_all(root: str, datasets=DATASETS, **kwargs):
    train, test = {}, {}
    for d in datasets:
        train[d], test[d] = load_dataset(d, root, **kwargs)
    return train, test
