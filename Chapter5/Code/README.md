# Optimising risk-based food safety inspection strategies

Code for Chapter 5 Optimising risk-based food safety inspection strategies.

- `ML_training/`: training of the CatBoost model that predicts compliance change
- `Convergence/`: determine number of simulation runs needed per strategy
- `Optimisation/`: optimisation of inspection strategies (all scenarios)

## Environment

```
conda create -n abm_env python=3.13
conda activate abm_env
pip install -r requirements.txt
```
`ML_training/` has its own `requirements.txt`.

## Data

Each folder contains the data and model files it needs.
