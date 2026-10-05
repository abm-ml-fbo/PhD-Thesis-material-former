"""
Optimisation of risk-based inspection strategies

Sobol sampling 200 strategies, top 40 as the initial population, GA for 30 generations,
top 3 strategies confirmed with 150 runs

"""
import os
import time
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from ml_inference import predict_inspection_result

# Settings
INPUT_FILE = "init_ABM_dataset.xlsx"
TICK_DAYS = 90
TICKS = 12
SIM_START_DATE = datetime.today().date()

DETERIORATION_STEP = 0.1
IMPROVEMENT_STEP = 1
WEIGHT_NAMES = ["w_size", "w_former", "w_type", "w_product", "w_interval"]
BOUNDS = (-4.0, 4.0)

# sensitivity analysis scenarios for Likelihood
LIKELIHOOD_SCHEMES = {
    "exponential": {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.8},
    "linear": {1: 0.1, 2: 0.33, 3: 0.57, 4: 0.8},
    "log": {1: 0.1, 2: 0.45, 3: 0.65, 4: 0.8},
}

N_SOBOL = 200
SOBOL_SEED = 42

# GA parameters
ELITISM = 3
TOURNAMENT_K = 3
P_MUT = 0.2
CX_ALPHA = 0.5   
GA_SEED = 2026

# Replications (early stop setting)
REP_SCHEDULE = (10, 20, 40)
FINAL_CONFIRM_REPS = 150
TOPK_FINAL = 3
BASE_SEED = 12345

# Logging
def setup_logging(output_dir):
    log_path = os.path.join(output_dir, "run_log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log

# Load data
def load_rows_with_date(path, n, sim_start_date, sample_seed=42):
    df = pd.read_excel(path, parse_dates=["Last Inspection date"], converters={"Product type": str})
    items = []
    for idx, r in df.sample(n, random_state=sample_seed).iterrows():
        last_date = r["Last Inspection date"].date()
        items.append({
            "agent_id": f"FBO_{idx}",
            "FBO type": r["FBO type"],
            "Product type": r["Product type"],
            "product_daly": float(r["Product type"]),
            "nr_employees": r["nr_employees"],
            "Operation type": r["Operation type"],
            "FBO Size": r["FBO Size"],
            "inspection_result": r["Inspection result"],
            "inspection_interval_days": max(0, (sim_start_date - last_date).days),
        })
    return items

# ABM
# FBO agents
class FBOAgent(Agent):
    def __init__(self, uid, model, info):
        super().__init__(uid, model)
        self.agent_id = info["agent_id"]
        self.FBO_type = info["FBO type"]
        self.Product_type = info["Product type"]
        self.product_daly = info["product_daly"]
        self.nr_employees = info["nr_employees"]
        self.Operation_type = info["Operation type"]
        self.FBO_size = info["FBO Size"]
        self.inspection_result = info["inspection_result"]

        self.compliance_level = self.inspection_result
        self.actual_compliance_level = self.compliance_level
        self.former_inspection_result = self.inspection_result
        self.inspection_interval_days = info["inspection_interval_days"]

    def features(self):
        return {
            "FBO type": self.FBO_type,
            "Product type": self.Product_type,
            "Operation type": self.Operation_type,
            "Former Inspection result": self.former_inspection_result,
            "FBO Size": self.FBO_size,
            "Inspection interval (days)": self.inspection_interval_days,
        }

    def pre_ml_tick_update(self):
        # possible deterioration, for FBOs not inspected in the most recent tick
        ch = predict_inspection_result(self.features())
        if ch == 1 and self.inspection_interval_days != TICK_DAYS:
            self.actual_compliance_level = min(4, self.actual_compliance_level + DETERIORATION_STEP)
            self.compliance_level = round(self.actual_compliance_level)

    def compute_risk(self):
        # risk index = likelihood x severity x exposure
        L = self.model.likelihood_map[self.compliance_level]
        S = self.product_daly
        E = np.log1p(self.nr_employees) / self.model.sector_max_log1p[self.Product_type]
        return L * S * E

    def inspect_and_maybe_improve(self):
        self.inspection_result = self.compliance_level
        ch = predict_inspection_result(self.features())
        if ch == -1:
            self.compliance_level = max(1, self.compliance_level - IMPROVEMENT_STEP)
        self.actual_compliance_level = self.compliance_level
        self.inspection_interval_days = TICK_DAYS
        self.former_inspection_result = self.inspection_result


# Inspector agents
class InspectorAgent(Agent):
    def __init__(self, uid, model, move_range, sight, inspection_capacity):
        super().__init__(uid, model)
        self.move_range = move_range
        self.sight = sight
        self.inspection_capacity = inspection_capacity

    def step(self):
        # random move, try to land on an empty cell
        for _ in range(12):
            dx = random.randint(-self.move_range, self.move_range)
            dy = random.randint(-self.move_range, self.move_range)
            nx = (self.pos[0] + dx) % self.model.grid.width
            ny = (self.pos[1] + dy) % self.model.grid.height
            if self.model.grid.is_cell_empty((nx, ny)):
                break
        self.model.grid.move_agent(self, (nx, ny))

        nb = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=self.sight)
        fbos = [a for a in nb if isinstance(a, FBOAgent)]
        if not fbos:
            return

        # rescale value of each factor to the same range
        def scale(arr):
            a = np.array(arr, dtype=float)
            mn, mx = a.min(), a.max()
            return np.full_like(a, 2.5) if mx == mn else 1 + 3 * (a - mn) / (mx - mn)

        sizes = scale([f.FBO_size for f in fbos])
        formers = scale([f.former_inspection_result for f in fbos])
        types = scale([f.FBO_type for f in fbos])
        products = scale([f.product_daly for f in fbos])
        intervals = scale([f.inspection_interval_days for f in fbos])

        w = self.model.weights
        score = (
            w["size"] * sizes
            + w["former"] * formers
            + w["type"] * types
            + w["product"] * products
            + w["interval"] * intervals
        )

        random_condition = self.model.random_inspection or all(v == 0 for v in w.values())
        order = np.random.permutation(len(fbos)) if random_condition else np.argsort(-score)

        selected = []
        for idx in order:
            f = fbos[idx]
            if f not in self.model.assigned_this_tick:
                selected.append(f)
                self.model.assigned_this_tick.add(f)
            if len(selected) >= self.inspection_capacity:
                break

        for f in selected:
            f.inspect_and_maybe_improve()

# Model
class InspectionModel(Model):
    def __init__(
        self, items, likelihood_map, num_inspectors, width, height,
        w_size=1.0, w_former=1.0, w_type=1.0, w_product=1.0, w_interval=1.0,
        move_range=18, sight=20, inspection_capacity=25, random_inspection=False,
    ):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        self.likelihood_map = likelihood_map
        self.weights = {
            "size": w_size, "former": w_former, "type": w_type,
            "product": w_product, "interval": w_interval
        }
        self.random_inspection = random_inspection
        self.assigned_this_tick = set()

        # FBOs
        for info in items:
            f = FBOAgent(info["agent_id"], self, info)
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(f, (x, y))
            self.schedule.add(f)

        # scale exposure to 0-1
        self.sector_max_log1p = {}
        for f in self.schedule.agents:
            val = np.log1p(f.nr_employees)
            self.sector_max_log1p[f.Product_type] = max(self.sector_max_log1p.get(f.Product_type, 0.0), val)

        # aggregated risk before any inspection (R0)
        self.total_risk_baseline = sum(f.compute_risk() for f in self.schedule.agents)
        self.effectiveness = 0.0

        # inspectors
        for i in range(num_inspectors):
            ins = InspectorAgent(f"INS_{i}", self, move_range, sight, inspection_capacity)
            while True:
                x, y = self.random.randrange(width), self.random.randrange(height)
                if self.grid.is_cell_empty((x, y)):
                    break
            self.grid.place_agent(ins, (x, y))
            self.schedule.add(ins)

    def step(self):
        self.assigned_this_tick.clear()

        # Pre-inspection update
        for a in self.schedule.agents:
            if isinstance(a, FBOAgent):
                a.pre_ml_tick_update()

        # Inspectors
        for a in self.schedule.agents:
            if isinstance(a, InspectorAgent):
                a.step()

        # aggregated risk (Rt) and effectiveness relative to R0
        self.total_risk_this_tick = sum(a.compute_risk() for a in self.schedule.agents if isinstance(a, FBOAgent))
        self.effectiveness = (self.total_risk_baseline - self.total_risk_this_tick) / self.total_risk_baseline

        # interval update for non-inspected
        for a in self.schedule.agents:
            if isinstance(a, FBOAgent) and a not in self.assigned_this_tick:
                a.inspection_interval_days += TICK_DAYS


# Define runs
def run_one_rep(items, w, seed, sim):
    random.seed(seed)
    np.random.seed(seed)
    model = InspectionModel(
        items,
        likelihood_map=sim["likelihood_map"],
        num_inspectors=sim["num_inspectors"],
        width=sim["width"],
        height=sim["height"],
        w_size=w["w_size"],
        w_former=w["w_former"],
        w_type=w["w_type"],
        w_product=w["w_product"],
        w_interval=w["w_interval"],
    )
    for _ in range(TICKS):
        model.step()
    return model.effectiveness * 100


def compute_stats(values):
    return np.mean(values), np.std(values, ddof=1)


def sequential_eval(uid, generation, stage, items, w, sim, log,
                    rep_schedule=REP_SCHEDULE, early_stop=True, base_seed=BASE_SEED):
    # add runs step by step, stop early if the mean effectiveness is negative
    start = time.time()
    vals = []
    status = "FULL_EVAL"
    for n in rep_schedule:
        while len(vals) < n:
            seed = base_seed + generation * 100000 + len(vals) * 17
            vals.append(run_one_rep(items, w, seed, sim))
        if early_stop and n < rep_schedule[-1] and np.mean(vals) < 0:
            status = "DISCARDED_EARLY"
            break

    mean, sd = compute_stats(vals)
    dt = time.time() - start
    log(f"[Eval] {stage} | gen={generation:02d} | {uid} | reps={len(vals):3d} | "
        f"mean={mean:8.3f}% | SD={sd:6.3f} | {status} | {dt:6.1f}s")

    return {
        "uid": uid, "generation": generation, "stage": stage,
        **w,
        "reps": len(vals), "mean": mean, "sd": sd, "sd_low": mean - sd, "sd_high": mean + sd,
        "status": status, "wall_time_sec": dt,
    }

# Sobol sampling initialise
def sobol_init_population(n=N_SOBOL, seed=SOBOL_SEED):
    sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
    x = qmc.scale(sampler.random(n=n), [BOUNDS[0]] * 5, [BOUNDS[1]] * 5)
    df = pd.DataFrame(x, columns=WEIGHT_NAMES)
    df.insert(0, "uid", [f"S{i + 1:03d}" for i in range(n)])
    return df

# GA operators
def tournament_select(pop, rng, k=TOURNAMENT_K):
    idx = rng.choice(pop.index.to_numpy(), size=k, replace=False)
    return pop.loc[idx].sort_values("fitness", ascending=False).iloc[0]


def blx_alpha(p1, p2, rng, alpha=CX_ALPHA):
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    r = hi - lo
    return lo - alpha * r + rng.random(p1.shape) * (r * (1 + 2 * alpha))


def mutate(x, sigma, rng, p=P_MUT):
    mask = rng.random(x.shape) < p
    x2 = x.copy()
    x2[mask] += rng.normal(0.0, sigma, size=x.shape)[mask]
    return np.clip(x2, BOUNDS[0], BOUNDS[1])


def sigma_at(gen, n_generations, sigma_start, sigma_end):
    if n_generations == 1:
        return sigma_end
    t = (gen - 1) / (n_generations - 1)
    return sigma_start + (sigma_end - sigma_start) * t

# Run
def main():
    parser = argparse.ArgumentParser(
        description="GA optimisation of risk-based inspection strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-n", type=int, default=1000, help="number of FBOs")
    parser.add_argument("--inspectors", type=int, default=8, help="number of inspectors")
    parser.add_argument("--grid-w", type=int, default=90, help="grid width")
    parser.add_argument("--grid-h", type=int, default=55, help="grid height")
    parser.add_argument("--likelihood", choices=list(LIKELIHOOD_SCHEMES), default="exponential",
                        help="likelihood scoring scheme")
    parser.add_argument("--pop-size", type=int, default=40, help="GA population size")
    parser.add_argument("--p-cx", type=float, default=0.9, help="crossover probability")
    parser.add_argument("--sigma-start", type=float, default=0.5, help="mutation sigma in the first generation")
    parser.add_argument("--sigma-end", type=float, default=0.1, help="mutation sigma in the last generation")
    parser.add_argument("--generations", type=int, default=30, help="number of GA generations")
    parser.add_argument("--output", type=str, default="results", help="output folder")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    log = setup_logging(output_dir)
    log(f"Start time: {datetime.now().isoformat()}")
    log(f"Settings: {vars(args)}")

    # scenario settings used in every run
    sim = {
        "likelihood_map": LIKELIHOOD_SCHEMES[args.likelihood],
        "num_inspectors": args.inspectors,
        "width": args.grid_w,
        "height": args.grid_h,
    }

    items = load_rows_with_date(INPUT_FILE, args.data_n, SIM_START_DATE)
    log(f"Loaded {len(items)} FBOs")

    eval_path = os.path.join(output_dir, "all_evaluations.csv")
    all_rows = []
    
    # Sobol sampling
    sobol_df = sobol_init_population()
    sobol_df.to_excel(os.path.join(output_dir, "sobol_initial_population.xlsx"), index=False)
    log(f"Evaluating {len(sobol_df)} Sobol samples, runs {REP_SCHEDULE}")

    for _, r in sobol_df.iterrows():
        w = {name: r[name] for name in WEIGHT_NAMES}
        row = sequential_eval(r["uid"], 0, "sobol_init", items, w, sim, log)
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(eval_path, index=False)

    sobol_res = pd.DataFrame(all_rows)
    sobol_res.to_excel(os.path.join(output_dir, "sobol_initial_results.xlsx"), index=False)

    # Initial GA population: best Sobol samples
    pop = sobol_res.sort_values("mean", ascending=False).reset_index(drop=True).head(args.pop_size)
    pop = pop.rename(columns={"mean": "fitness"})
    pop[["uid", *WEIGHT_NAMES, "fitness", "reps", "sd_low", "sd_high", "status"]].to_excel(
        os.path.join(output_dir, "ga_init_population.xlsx"), index=False)
    log(f"Initial population: top {args.pop_size} Sobol samples")
    
    # GA
    rng = np.random.default_rng(GA_SEED)
    gen_summaries = []
    best_trace = []

    for gen in range(1, args.generations + 1):
        sigma = sigma_at(gen, args.generations, args.sigma_start, args.sigma_end)

        # current population
        fitness = pop["fitness"].astype(float)
        gen_summaries.append({
            "generation": gen, "sigma": sigma,
            "best_fitness": fitness.max(), "mean_fitness": fitness.mean(), "std_fitness": fitness.std(ddof=1),
            "early_stopped": (pop["status"] == "DISCARDED_EARLY").sum(),
            "full_eval": (pop["status"] == "FULL_EVAL").sum(),
        })
        best_trace.append({"generation": gen, "best_fitness": fitness.max()})
        log(f"Generation {gen}/{args.generations} | sigma={sigma:.3f} | "
            f"best={fitness.max():.3f}% | mean={fitness.mean():.3f}%")

        # keep the best strategies
        elites = pop.sort_values("fitness", ascending=False).head(ELITISM)

        # new strategies from selected parents
        children = []
        need = args.pop_size - ELITISM
        while len(children) < need:
            x1 = tournament_select(pop, rng)[WEIGHT_NAMES].to_numpy(dtype=float)
            x2 = tournament_select(pop, rng)[WEIGHT_NAMES].to_numpy(dtype=float)
            if rng.random() < args.p_cx:
                c1, c2 = blx_alpha(x1, x2, rng), blx_alpha(x2, x1, rng)
            else:
                c1, c2 = x1.copy(), x2.copy()
            c1, c2 = mutate(c1, sigma, rng), mutate(c2, sigma, rng)
            for c in (c1, c2):
                if len(children) < need:
                    children.append({"uid": f"G{gen:02d}_C{len(children) + 1:03d}", **dict(zip(WEIGHT_NAMES, c))})

        # evaluate the new strategies
        evaluated = []
        for child in children:
            w = {name: child[name] for name in WEIGHT_NAMES}
            row = sequential_eval(child["uid"], gen, "ga", items, w, sim, log)
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(eval_path, index=False)
            evaluated.append({**child, "fitness": row["mean"], "reps": row["reps"],
                              "sd_low": row["sd_low"], "sd_high": row["sd_high"], "status": row["status"]})

        pop = pd.concat([elites, pd.DataFrame(evaluated)], ignore_index=True)

        pd.DataFrame(gen_summaries).to_excel(os.path.join(output_dir, "ga_generation_summary.xlsx"), index=False)
        pd.DataFrame(best_trace).to_csv(os.path.join(output_dir, "ga_best_trace.csv"), index=False)
        pop.to_excel(os.path.join(output_dir, "ga_current_population.xlsx"), index=False)
    
     # top 3 strategies 150 runs
    top = pd.DataFrame(all_rows).sort_values("mean", ascending=False).head(TOPK_FINAL)
    final_rows = []
    for _, r in top.iterrows():
        w = {name: r[name] for name in WEIGHT_NAMES}
        row = sequential_eval(f"FINAL_{r['uid']}", args.generations + 1, "final", items, w, sim, log,
                              rep_schedule=(FINAL_CONFIRM_REPS,), early_stop=False, base_seed=BASE_SEED + 999999)
        final_rows.append(row)
    pd.DataFrame(final_rows).sort_values("mean", ascending=False).to_excel(
        os.path.join(output_dir, "final_top3_confirmed.xlsx"), index=False)

    # Plot
    bt = pd.DataFrame(best_trace)
    plt.figure(figsize=(8, 5))
    plt.plot(bt["generation"], bt["best_fitness"], linewidth=2, marker="o", markersize=4)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Best Fitness (effectiveness %)", fontsize=12)
    plt.title("GA Optimization Progress", fontsize=13, fontweight="bold")
    plt.grid(True, linewidth=0.3, alpha=0.7)
    plt.savefig(os.path.join(output_dir, "fitness_evolution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    log(f"End time: {datetime.now().isoformat()}")
    log(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()
