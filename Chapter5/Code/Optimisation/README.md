# Optimisation

`optimisation_run.py` samples 200 strategies with Sobol sampling, uses the best 40 as the initial population of a genetic algorithm (30 generations), and confirms the top 3 strategies with 150 runs.

## Run

Upload all files in this folder to the same folder on the HPC, then submit one job per scenario with Slurm from that folder:

```
sbatch [Slurm options] job.sh [scenario options]
```

| Scenario | Command |
|---|---|
| Baseline | `sbatch --job-name=baseline job.sh --output results_baseline` |
| 6 inspectors | `sbatch --job-name=6_inspector job.sh --inspectors 6 --output results_6_inspector` |
| 10 inspectors | `sbatch --job-name=10_inspector job.sh --inspectors 10 --output results_10_inspector` |
| Population 36 | `sbatch --job-name=population_36 job.sh --pop-size 36 --output results_population_36` |
| Population 44 | `sbatch --job-name=population_44 job.sh --pop-size 44 --output results_population_44` |
| Crossover 0.81 | `sbatch --job-name=crossover_0.81 job.sh --p-cx 0.81 --output results_crossover_0.81` |
| Crossover 0.99 | `sbatch --job-name=crossover_0.99 job.sh --p-cx 0.99 --output results_crossover_0.99` |
| Sigma 0.45–0.09 | `sbatch --job-name=sigma_0.45_0.09 job.sh --sigma-start 0.45 --sigma-end 0.09 --output results_sigma_0.45_0.09` |
| Sigma 0.55–0.11 | `sbatch --job-name=sigma_0.55_0.11 job.sh --sigma-start 0.55 --sigma-end 0.11 --output results_sigma_0.55_0.11` |
| Likelihood linear | `sbatch --job-name=likelihood_linear job.sh --likelihood linear --output results_likelihood_linear` |
| Likelihood log | `sbatch --job-name=likelihood_log job.sh --likelihood log --output results_likelihood_log` |
| Scale small | `sbatch --job-name=scale_small job.sh --data-n 500 --inspectors 4 --grid-w 64 --grid-h 39 --output results_scale_small` |
| Scale large | `sbatch --job-name=scale_large --time=8-10:00:00 job.sh --data-n 1500 --inspectors 12 --grid-w 110 --grid-h 68 --output results_scale_large` |


### Start date

The simulation starts on the date it is run, and the inspection interval of each FBO is the number of days between its last inspection date and this start date. To use the start date of the thesis (3 October 2026), replace the `python` line in `job.sh` with:

```
python -c "import sys, datetime, optimisation_run as o; o.SIM_START_DATE = datetime.date(2026, 10, 3); sys.argv = ['optimisation_run.py'] + sys.argv[1:]; o.main()" "$@"
```

The scenario options in the table are used in the same way.

## Output

The results of each scenario are saved in the folder given by `--output`.
