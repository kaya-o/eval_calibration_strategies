import csv
from datetime import datetime
from pathlib import Path
import numpy as np

N_OFF = 10
N_ON = 20
TAU_0 = 20
TAU_1 = 16
BETA = 1
ALPHA = 0.4


def mu(x, beta=BETA):
        return x*beta

def generate_datapoints(n):
    rng = np.random.default_rng()

    # X ~ Uniform[0, 2]
    X = rng.uniform(0.0, 2.0, size=n)

    # epsilon|X ~ N(0, X/2)
    # np.random.normal uses standard deviation, not variance
    epsilon_std = np.sqrt(X / 2.0)
    epsilon = rng.normal(loc=0.0, scale=epsilon_std)
    
    Y = X + epsilon
    scores = np.abs(mu(X) - Y)
    return X, Y, scores


def dump_experiment_results(results, raw_rows, n_runs, output_root="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"{timestamp}_{n_runs}_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_dir / "aggregate_results.csv"
    raw_path = output_dir / "raw_selected_events.csv"

    aggregate_fieldnames = [
        "strategy",
        "selected",
        "miscovered",
        "miscoverage",
        "avg_n_calibration",
        "median_interval_length",
        "infinite_fraction",
    ]

    with aggregate_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fieldnames)
        writer.writeheader()

        for strategy, strategy_results in results.items():
            selected = strategy_results["selected"]
            if selected == 0:
                writer.writerow({
                    "strategy": strategy,
                    "selected": 0,
                    "miscovered": 0,
                    "miscoverage": np.nan,
                    "avg_n_calibration": np.nan,
                    "median_interval_length": np.nan,
                    "infinite_fraction": np.nan,
                })
                continue

            writer.writerow({
                "strategy": strategy,
                "selected": selected,
                "miscovered": strategy_results["miscovered"],
                "miscoverage": strategy_results["miscovered"] / selected,
                "avg_n_calibration": np.mean(strategy_results["n_calibration"]),
                "median_interval_length": np.median(strategy_results["interval_length"]),
                "infinite_fraction": strategy_results["infinite_interval"] / selected,
            })

    raw_fieldnames = [
        "run",
        "t",
        "strategy",
        "miscovered",
        "n_calibration",
        "interval_length",
        "threshold",
        "score_t",
        "sum_s_past",
    ]

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    return output_dir

class Conformal:
    def __init__(self):
        self.x_off = np.array([])
        self.y_off = np.array([])
        self.scores_off = np.array([])
        self.x_past = np.array([])
        self.y_past = np.array([])
        self.scores_past = np.array([])
        self.s_past = np.array([])

    def select_past(self, x, j, tau_0=TAU_0):
        return int(x < 1 + ((1/tau_0)*sum(self.s_past[:j])))

    def select_t(self, x=None, j=None, tau_1=TAU_1):
        s_past = [self.select_past(x_i, i) for i, x_i in enumerate(self.x_past)]
        return int(np.sum(s_past) > tau_1)

    def quantile_threshold(self, calibration_scores, alpha=ALPHA):
        if len(calibration_scores) == 0:
            return np.inf

        sorted_scores = np.sort(calibration_scores)
        n = len(sorted_scores)

        quantile_idx = int(np.ceil(len(sorted_scores) * (1 - alpha))) - 1

        if quantile_idx >= n:
            return np.inf

        return sorted_scores[quantile_idx]

    # Calibration strategies
    def full(self):
        return np.concatenate([self.x_off, self.x_past]), np.concatenate([self.y_off, self.y_past])

    def s_full(self):
        x_candidates = np.concatenate([self.x_off, self.x_past])
        y_candidates = np.concatenate([self.y_off, self.y_past])
        selected_mask = np.array([self.select_t(x) == 1 for x in x_candidates])
        x_selected = x_candidates[selected_mask]
        y_selected = y_candidates[selected_mask]
        return x_selected, y_selected

    def s_fix(self):
        selected_mask = np.array([self.select_t(x) == 1 for x in self.x_off])
        x_selected = self.x_off[selected_mask]
        y_selected = self.y_off[selected_mask]
        return x_selected, y_selected

    def s_fix_hack(self):
        threshold = 1 + ((1 / TAU_0) * np.sum(self.s_past))
        selected_mask = np.array([x_j < threshold for x_j in self.x_off])
        x_selected = self.x_off[selected_mask]
        y_selected = self.y_off[selected_mask]
        return x_selected, y_selected

    # bao et al. 2024
    def ada_off(self):
        selected_mask = np.array([self.select_t(x_j) == 1 for x_j in self.x_off])
        return self.x_off[selected_mask], self.y_off[selected_mask]

    def ada_on(self, x_t):
        selected_mask = np.array([self.select_t(x_j) == 1 and self.select_past(x_j, j) == self.select_past(x_t, j) for j,x_j in enumerate(self.x_past)])
        return self.x_past[selected_mask], self.y_past[selected_mask]

    def express(self, x_t):
        x_candidates = np.concatenate([self.x_off, self.x_past])
        y_candidates = np.concatenate([self.y_off, self.y_past])
        selected_mask = np.array([
            bool(self.select_t(x_j) 
             * np.prod([int(self.select_past(x_j, i) == self.select_past(x_t, i)) for i, x_i in enumerate(self.x_past)])) 
                for x_j in x_candidates
        ])
        return x_candidates[selected_mask], y_candidates[selected_mask]
    
    def k_express(self, x_t, k):
        x_candidates = np.concatenate([self.x_off, self.x_past[-k:]])
        y_candidates = np.concatenate([self.y_off, self.y_past[-k:]])
        rule_idx = range(len(self.x_past)-k, len(self.x_past))
        selected_mask = np.array([
            bool(self.select_t(x_j) 
             * np.prod([int(self.select_past(x_j, i) == self.select_past(x_t, i)) for i in rule_idx])) 
                for x_j in x_candidates
        ])
        return x_candidates[selected_mask], y_candidates[selected_mask]
    
    def express_m(self, x_t, k):
        t = len(self.x_past)
        if t == 0:
            return np.inf

        alpha_sf = (1 / np.sqrt(t)) * ALPHA
        alpha_ex = (1 - (1 / np.sqrt(t))) * ALPHA

        x_sf, y_sf = self.s_fix()
        scores_sf = self.compute_scores(x_sf, y_sf)
        threshold_sf = self.quantile_threshold(scores_sf, alpha=alpha_sf)

        x_ex, y_ex = self.express(x_t)
        scores_ex = self.compute_scores(x_ex, y_ex)
        threshold_ex = self.quantile_threshold(scores_ex, alpha=alpha_ex)

        return min(threshold_sf, threshold_ex)

    def compute_scores(self, x, y):
        return np.abs(mu(x) - y)

    def append_online_point(self, x_t, y_t, score_t, s_t):
        self.x_past = np.append(self.x_past, x_t)
        self.y_past = np.append(self.y_past, y_t)
        self.scores_past = np.append(self.scores_past, score_t)
        self.s_past = np.append(self.s_past, s_t)

    def evaluate_strategy(self, strategy, x_t, y_t, k=5):
        if strategy == "FULL":
            x_cal, y_cal = self.full()
        elif strategy == "S-FULL":
            x_cal, y_cal = self.s_full()
        elif strategy == "S-FIX":
            x_cal, y_cal = self.s_fix()
        elif strategy == "ADA":
            x_off, y_off = self.ada_off()
            x_on, y_on = self.ada_on(x_t)
            x_cal = np.concatenate([x_off, x_on])
            y_cal = np.concatenate([y_off, y_on])
        elif strategy == "EXPRESS":
            x_cal, y_cal = self.express(x_t)
        elif strategy == "K-EXPRESS":
            x_cal, y_cal = self.k_express(x_t, k)
        elif strategy == "EXPRESS-M":
            threshold = self.express_m(x_t, k)
            x_sf, y_sf = self.s_fix()
            x_ex, y_ex = self.express(x_t)
            n_calibration = len(x_sf) + len(x_ex)
            score_t = abs(mu(x_t) - y_t)
            covered = score_t <= threshold

            return {
                "miscovered": not covered,
                "n_calibration": n_calibration,
                "interval_length": 2 * threshold,
                "threshold": threshold,
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        calibration_scores = self.compute_scores(x_cal, y_cal)
        threshold = self.quantile_threshold(calibration_scores)
        score_t = abs(mu(x_t) - y_t)
        covered = score_t <= threshold

        return {
            "miscovered": not covered,
            "n_calibration": len(calibration_scores),
            "interval_length": 2 * threshold,
            "threshold": threshold,
        }

if __name__ == "__main__":
    strategies = ["FULL", "S-FULL", "S-FIX", "ADA", "EXPRESS", "K-EXPRESS", "EXPRESS-M"]
    results = {
        strategy: {
            "selected": 0,
            "miscovered": 0,
            "n_calibration": [],
            "interval_length": [],
            "infinite_interval": 0,
        }
        for strategy in strategies
    }

    total_hits = 0
    n_runs = (10**4)*5
    raw_rows = []

    for run in range(n_runs):
        if (run % 1000 == 0):
            print(f"Run: {run}")
        conformal = Conformal()
        x_all, y_all, scores_all = generate_datapoints(N_ON + N_OFF)
        conformal.x_off = x_all[:N_OFF]
        conformal.y_off = y_all[:N_OFF]
        conformal.scores_off = scores_all[:N_OFF]

        x_on = x_all[N_OFF:]
        y_on = y_all[N_OFF:]
        scores_on = scores_all[N_OFF:]

        for t in range(N_ON):
            x_t = x_on[t]
            y_t = y_on[t]
            score_t = scores_on[t]

            # Current selected/reported point under the second branch.
            s_t = conformal.select_t(x_t, t)

            if s_t:
                total_hits += 1

                for strategy in strategies:
                    strategy_result = conformal.evaluate_strategy(strategy, x_t, y_t, k=5)

                    results[strategy]["selected"] += 1
                    results[strategy]["miscovered"] += int(strategy_result["miscovered"])
                    results[strategy]["n_calibration"].append(strategy_result["n_calibration"])
                    results[strategy]["interval_length"].append(strategy_result["interval_length"])
                    results[strategy]["infinite_interval"] += int(np.isinf(strategy_result["interval_length"]))
                    raw_rows.append({
                        "run": run,
                        "t": t,
                        "strategy": strategy,
                        "miscovered": int(strategy_result["miscovered"]),
                        "n_calibration": strategy_result["n_calibration"],
                        "interval_length": strategy_result["interval_length"],
                        "threshold": strategy_result["threshold"],
                        "score_t": score_t,
                        "sum_s_past": np.sum(conformal.s_past),
                    })

            # Historical value used by the first branch for future times.
            s_history_t = conformal.select_past(x_t, t)
            conformal.append_online_point(x_t, y_t, score_t, s_history_t)

    print(f"total_hits={total_hits}")

    for strategy in strategies:
        selected = results[strategy]["selected"]
        if selected == 0:
            print(f"{strategy}: no selected points")
            continue

        miscoverage = results[strategy]["miscovered"] / selected
        avg_n_calibration = np.mean(results[strategy]["n_calibration"])
        median_interval_length = np.median(results[strategy]["interval_length"])
        infinite_fraction = results[strategy]["infinite_interval"] / selected

        print(
            f"{strategy}: "
            f"selected={selected}, "
            f"miscoverage={miscoverage:.3f}, "
            f"avg_n_calibration={avg_n_calibration:.2f}, "
            f"median_interval_length={median_interval_length:.3f}, "
            f"infinite_fraction={infinite_fraction:.3f}"
        )

    output_dir = dump_experiment_results(results, raw_rows, n_runs)
    print(f"Wrote results to {output_dir}")
