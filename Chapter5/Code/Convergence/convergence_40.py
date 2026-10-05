"""
Convergence test with early stopping - ABM

Sobol sampling 40 samples, 10 initial runs per sample, 100 runs if not stopped early

"""
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import qmc

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from ml_inference import predict_inspection_result

# Settings
INPUT_FILE = "init_ABM_dataset.xlsx"
OUTPUT_DIR = "results_40"
GRID_W, GRID_H = 90, 55
TICK_DAYS = 90
SIM_START_DATE = datetime.today().date()

LIKELIHOOD_MAP = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.8}
DETERIORATION_STEP = 0.1
IMPROVEMENT_STEP = 1
WEIGHT_NAMES = ["w_size", "w_former", "w_type", "w_product", "w_interval"]

N_SAMPLES = 40
LOW, HIGH = -4.0, 4.0 
SOBOL_SEED = 42
BASE_SEED = 12345

N_TICKS = 12
N_FBO = 1000
N_INIT = 10
N_FULL = 100

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
        self.actual_compliance_level = self.compliance_level   # latent continuous score
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
        L = LIKELIHOOD_MAP[self.compliance_level]
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

        # all weights 0 is treated as random inspection
        random_condition = self.model.random_inspection or all(v == 0 for v in w.values())
        order = np.random.permutation(len(fbos)) if random_condition else np.argsort(-score)

        # Select FBOs to be inspected
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
        self, items, num_inspectors=8, width=GRID_W, height=GRID_H,
        w_size=1.0, w_former=1.0, w_type=1.0, w_product=1.0, w_interval=1.0,
        move_range=18, sight=20, inspection_capacity=25, random_inspection=False,
    ):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

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

        self.datacollector = DataCollector(model_reporters={"effectiveness": lambda m: m.effectiveness})

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

        self.datacollector.collect(self)

# Define runs
def get_final_effectiveness_pct(model):
    df = model.datacollector.get_model_vars_dataframe()
    return df["effectiveness"].iloc[-1] * 100


def run_single_sim(weights, items, n_ticks):
    model = InspectionModel(
        items,
        w_size=weights["w_size"],
        w_former=weights["w_former"],
        w_type=weights["w_type"],
        w_product=weights["w_product"],
        w_interval=weights["w_interval"],
    )
    for _ in range(n_ticks):
        model.step()
    return get_final_effectiveness_pct(model)


def replicate(weights, n_reps, items, sample_id, first_rep=0):
    results = []
    for rep in range(first_rep, first_rep + n_reps):
        seed = BASE_SEED + sample_id * 1000 + rep
        random.seed(seed)
        np.random.seed(seed)
        results.append(run_single_sim(weights, items, N_TICKS))
    return np.array(results)


def compute_stats(values):
    return np.mean(values), np.std(values, ddof=1)


def running_stats(values):
    means = [np.mean(values[:k]) for k in range(1, len(values) + 1)]
    sds = [0.0] + [np.std(values[:k], ddof=1) for k in range(2, len(values) + 1)]
    return np.array(means), np.array(sds)

# Run
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sobol sampling
    sampler = qmc.Sobol(d=5, scramble=True, seed=SOBOL_SEED)
    x = qmc.scale(sampler.random(n=N_SAMPLES), [LOW] * 5, [HIGH] * 5)
    weights_df = pd.DataFrame(x, columns=WEIGHT_NAMES)
    weights_df.insert(0, "sample_id", np.arange(1, len(weights_df) + 1))
    weights_df.to_excel(os.path.join(OUTPUT_DIR, f"sobol_{N_SAMPLES}_weights.xlsx"), index=False)

    items = load_rows_with_date(INPUT_FILE, N_FBO, SIM_START_DATE)

    summary_rows = []
    raw_rows = []
    all_sample_data = {}

    for _, row in weights_df.iterrows():
        sample_id = int(row["sample_id"])
        weights = {name: row[name] for name in WEIGHT_NAMES}

        # first 10 runs, stop early if the mean effectiveness is negative
        init_vals = replicate(weights, N_INIT, items, sample_id)
        init_mean, init_sd = compute_stats(init_vals)
        if init_mean < 0:
            status = "DISCARDED_EARLY"
            results = init_vals
        else:
            status = "FULL_EVAL"
            extra_vals = replicate(weights, N_FULL - N_INIT, items, sample_id, first_rep=N_INIT)
            results = np.concatenate([init_vals, extra_vals])
        print(f"Sample {sample_id}/{N_SAMPLES} | {status} | {len(results)} runs", flush=True)

        for i, val in enumerate(results, start=1):
            raw_rows.append({"sample_id": sample_id, "rep": i, "final_effectiveness_pct": val, "status": status})

        mean, sd = compute_stats(results)
        summary_rows.append({
            "sample_id": sample_id,
            **weights,
            "status": status,
            "n_reps_used": len(results),
            "mean_effectiveness_pct": mean,
            "std_dev": sd,
            "sd_lower_bound": mean - sd,
            "sd_upper_bound": mean + sd,
            "init_mean_10": init_mean,
            "init_sd_10": init_sd,
        })
        means, sds = running_stats(results)
        all_sample_data[sample_id] = (status, means, sds, mean, sd)

    all_lower = np.concatenate([means - sds for _, means, sds, _, _ in all_sample_data.values()])
    all_means = np.concatenate([means for _, means, _, _, _ in all_sample_data.values()])
    pad = 0.1 * (all_means.max() - all_lower.min())
    y_min, y_max = all_lower.min() - pad, all_means.max() + pad

    pdf_path = os.path.join(OUTPUT_DIR, f"sequential_{N_SAMPLES}_convergence.pdf")
    with PdfPages(pdf_path) as pdf:
        for sample_id, (status, means, sds, mean, sd) in all_sample_data.items():
            runs = np.arange(1, len(means) + 1)
            plt.figure(figsize=(7.2, 4.0))
            plt.plot(runs, means, linewidth=1.8, color="steelblue", label="Mean")
            plt.fill_between(runs, means - sds, means + sds, alpha=0.2, color="steelblue", label="±1 SD")
            plt.axhline(0, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
            plt.title(f"Sample {sample_id} | {status} | reps={len(means)}\nMean={mean:.2f}%, SD={sd:.2f}%", fontsize=11)
            plt.xlabel("Number of replications", fontsize=10)
            plt.ylabel("Final effectiveness (%)", fontsize=10)
            plt.ylim(y_min, y_max)
            plt.legend(loc="best", fontsize=9)
            plt.grid(True, alpha=0.25, linestyle="--")
            plt.tight_layout()
            pdf.savefig(dpi=100)
            plt.close()

    summary_df = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, f"sequential_{N_SAMPLES}_mean_sd.xlsx")) as writer:
        weights_df.to_excel(writer, sheet_name="weights", index=False)
        summary_df.to_excel(writer, sheet_name="mean_sd_table", index=False)
        pd.DataFrame(raw_rows).to_excel(writer, sheet_name="raw_results", index=False)

    best = summary_df.loc[summary_df["mean_effectiveness_pct"].idxmax()]
    print(f"Early stopped: {(summary_df['status'] == 'DISCARDED_EARLY').sum()} | full: {(summary_df['status'] == 'FULL_EVAL').sum()}")
    print(f"Best sample: {int(best['sample_id'])} | mean effectiveness: {best['mean_effectiveness_pct']:.3f}%")
    print(f"Results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()