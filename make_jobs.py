from aura.traces import DATASETS

STEPS = 600000
PURE_STEPS = 1000000


def job(name, sources, seed, extra="", steps=STEPS):
    return (f"python3 train.py --sources {','.join(sources)} --seed {seed} --timesteps {steps} "
            f"{extra} --out runs/{name}_s{seed} > logs/{name}_s{seed}.log 2>&1")


def lodo(seed):
    return [job(f"lodo_{d}", [x for x in DATASETS if x != d], seed, "--mode residual")
            for d in DATASETS]


def main():
    batch1 = [job("all", DATASETS, s, "--mode residual") for s in (0, 1, 2)]
    batch1 += [job("all_noctx", DATASETS, s, "--mode residual --no-context") for s in (0, 1)]
    batch1 += lodo(0)
    batch1 += [job(f"all_delay{d}", DATASETS, 0, f"--mode residual --migration-delay {d}")
               for d in (0.0, 2.0, 4.0)]
    batch1 += [job("all_pure", DATASETS, 0, "--mode factored --gamma 0.99", PURE_STEPS)]
    batch2 = [job("all_noctx", DATASETS, 2, "--mode residual --no-context")]
    batch2 += lodo(1)
    batch2 += [job(f"all_delay{d}", DATASETS, 1, f"--mode residual --migration-delay {d}")
               for d in (0.0, 2.0, 4.0)]
    batch2 += [job("all_pure", DATASETS, s, "--mode factored --gamma 0.99", PURE_STEPS)
               for s in (1, 2)]
    batch3 = [job("lodo_5g", [x for x in DATASETS if x != "5g"], 2, "--mode residual")]
    for name, jobs in (("jobs_batch1.txt", batch1), ("jobs_batch2.txt", batch2),
                       ("jobs_batch3.txt", batch3)):
        with open(name, "w") as f:
            f.write("\n".join(jobs) + "\n")
        print(name, len(jobs))


if __name__ == "__main__":
    main()
