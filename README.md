# AURA-Stream

Joint bitrate and edge selection for video streaming, evaluated by replaying real
throughput traces from 3G to commercial 5G. The repository contains the simulator,
heuristic and learned controllers, and the scripts that produce every result.

## Setup

```bash
pip install -r requirements.txt
bash scripts/fetch_data.sh data
```

`fetch_data.sh` downloads two public collections into `data/`:

- [Real-world bandwidth traces](https://github.com/confiwent/Real-world-bandwidth-traces):
  HSDPA, FCC, Oboe and Puffer throughput traces
- [5G production dataset](https://github.com/uccmisl/5Gdataset): 5G/LTE traces with
  speed, RSRP, cell ID and ping latency

Set `AURA_DATA` or pass `--data-root` if the data lives elsewhere.

## Layout

| Path | Contents |
|------|----------|
| `aura/traces.py` | Dataset loading, per-file train/test split, 300 s segments |
| `aura/env.py` | Time-based replay with radio/edge bottleneck and migration delay |
| `aura/baselines.py` | Heuristic controllers (rate, buffer, context-aware, RobustMPC-style) |
| `aura/advisors.py` | Action spaces for learning on top of a model-based planner |
| `aura/gym_env.py` | Gymnasium wrapper used for training |
| `aura/evaluation.py` | Evaluation with common random numbers, paired bootstrap |
| `train.py` | Train a PPO or DQN agent |
| `evaluate.py` | Evaluate agents and baselines on the test splits |
| `analyze.py` | Tables and statistics from the evaluation CSVs |
| `make_jobs.py` | The list of training runs reported in the paper |
| `figures/` | Figure scripts |
| `scripts/` | Data download, handover-discount ablation, action statistics |
| `results/` | Evaluation CSVs and summary tables |

## Reproducing the results

`results/` contains the evaluation CSVs behind the paper. To regenerate them, write
into a fresh directory (here `out/`) so the shipped files are not counted twice. The
steps are: baselines at every migration delay, the training runs (three batches, one
run per line), evaluation of all trained agents, the ablations, and the tables and
figures.

```bash
mkdir -p out figures/out logs runs

for d in 0.0 1.0 2.0 4.0; do
  python evaluate.py --baselines --migration-delay $d --out out/baselines_d$d.csv
done

python make_jobs.py
for b in 1 2 3; do xargs -P 2 -I CMD bash -c CMD < jobs_batch$b.txt; done

python evaluate.py --runs runs/* --out out/agent_all_runs.csv
python scripts/handover_ablation.py --out out/handover_ablation.csv
python scripts/action_stats.py --runs runs/all_s0 runs/all_s1 runs/all_s2 \
    runs/lodo_5g_s0 runs/lodo_5g_s1 runs/lodo_5g_s2 --dataset 5g \
    --out out/action_stats_5g.csv
python analyze.py --results out
python figures/results.py --results out --out figures/out
python figures/architecture.py figures/out/fig_architecture
```

A single run, for example learning on top of the MPC planner with all five
datasets:

```bash
python train.py --sources hsdpa,fcc,oboe,puffer,5g --mode residual \
    --timesteps 600000 --seed 0 --out runs/all_s0
```

`--mode` selects the action space: `factored` (bitrate and edge chosen directly),
`residual` (edge plus a bitrate correction to the MPC plan), `residual_ctx`
(correction to the context-aware rule), or `experts` (choose which controller to
follow). Other options: `--no-context`, `--migration-delay`, `--algo DQN`.
