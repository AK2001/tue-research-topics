import numpy as np
import gymnasium as gym
import d3rlpy
import time


def subsample(dataset, fraction, seed=0):
    episodes = dataset.episodes
    n_keep = max(1, int(len(episodes) * fraction))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(episodes), size=n_keep, replace=False)
    return d3rlpy.dataset.ReplayBuffer(
        d3rlpy.dataset.InfiniteBuffer(),
        episodes=[episodes[i] for i in idx],
    )


def evaluate(algo, n_episodes=50):
    eval_env = gym.make("Pendulum-v1")
    scores = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total = 0.0
        while not done:
            action = algo.predict(np.expand_dims(obs, axis=0))[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    return scores


DATA_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2]
N_STEPS = 50000
DEVICE = "mps"            # or "mps" if your speed test showed it's faster

total_runs = len(DATA_LEVELS) * len(SEEDS)
run_number = 0
sweep_start = time.time()

full_dataset, _ = d3rlpy.datasets.get_pendulum()

results = {}

for fraction in DATA_LEVELS:
    per_seed_means = []            # one average score per seed
    all_scores_this_level = []     # every episode across every seed

    for seed in SEEDS:
        run_number += 1
        elapsed = time.time() - sweep_start
        avg_per_run = elapsed / (run_number - 1) if run_number > 1 else 0
        remaining = avg_per_run * (total_runs - run_number + 1)

        print(f"[{run_number}/{total_runs} {100*run_number/total_runs:.0f}%] "
              f"data={int(fraction*100)}%, seed={seed} | "
              f"elapsed {elapsed/60:.1f} min, "
              f"~{remaining/60:.1f} min left")
        
        d3rlpy.seed(seed)          # fix the randomness for this run
        data = subsample(full_dataset, fraction)
        algo = d3rlpy.algos.TD3PlusBCConfig().create(device=DEVICE)
        algo.fit(data, n_steps=N_STEPS, show_progress=True)

        scores = evaluate(algo, n_episodes=50)
        per_seed_means.append(np.mean(scores))
        all_scores_this_level.extend(scores)

    all_scores_this_level = np.array(all_scores_this_level)
    results[fraction] = {
        "mean_of_seed_means": np.mean(per_seed_means),
        "std_across_seeds":  np.std(per_seed_means),
        "median":            np.median(all_scores_this_level),
        "worst":             np.min(all_scores_this_level),
    }

# print a clean summary table
print(f"{'data%':>6} | {'mean':>9} | {'±seed std':>9} | {'median':>9} | {'worst':>9}")
print("-" * 55)
for fraction in DATA_LEVELS:
    r = results[fraction]
    print(f"{int(fraction*100):>5}% | {r['mean_of_seed_means']:>9.1f} | "
          f"{r['std_across_seeds']:>9.1f} | {r['median']:>9.1f} | {r['worst']:>9.1f}")