# Convergence

- `convergence_10.py`: 10 Sobol samples, 150 runs each
- `convergence_40.py`: 40 Sobol samples, 10 runs each, continued to 100 runs if the mean effectiveness is not negative

## Run

Upload all files in this folder to the same folder on the HPC, then submit the jobs with Slurm:

```
sbatch job_con10.sh
sbatch job_con40.sh
```


### Start date

The simulation starts on the date it is run, and the inspection interval of each FBO is the number of days between its last inspection date and this start date. To use the start date of the thesis (3 October 2026), replace the `python` line in the job scripts.

In `job_con10.sh`:

```
python -c "import sys, datetime, convergence_10 as c; c.SIM_START_DATE = datetime.date(2026, 10, 3); sys.argv = ['convergence_10.py'] + sys.argv[1:]; c.main()" \
    --n-samples 10 --n-reps 150 --n-ticks 12 --n-workers 32 --n-fbo 1000 --output results/
```

In `job_con40.sh`:

```
python -c "import datetime, convergence_40 as c; c.SIM_START_DATE = datetime.date(2026, 10, 3); c.main()"
```

## Output

`convergence_10.py` saves its results in `results/`, and `convergence_40.py` in `results_40/`.
