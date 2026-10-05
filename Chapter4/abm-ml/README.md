# ML-ABM framework for evaluating different risk-based inspection strategies

Code and data for Chapter 4

## Files

| File | Description |
|------|-------------|
| `1_train_ml_model.ipynb` | Trains catboost classifier |
| `2_abm_demo.ipynb` | Interactive ABM demo |
| `3_scenarios_and_sensitivity.ipynb` | Scenarios and sensitivity analysis (results in the paper) |
| `ml_inference.py` | Catboost model predictions (-1 / 0 / +1) |
| `final_catboost_model.cbm`, `final_catboost_model.meta.json` | Trained model and its metadata |
| `ml_train_dataset.xlsx` | Training data (41,525 records) |
| `init_ABM_dataset.xlsx` | ABM initialisation data (37,867 FBOs) |
| `requirements.txt` | Required packages |


## Usage

Python 3.10 or 3.11. Keep all files in one folder.

    pip install -r requirements.txt
    jupyter lab

1. `1_train_ml_model.ipynb` (optional): run it will retrain and overwrite the current model files.
2. `2_abm_demo.ipynb`: run all cells, press *Initialize*, then *Step* or *Play*.
3. `3_scenarios_and_sensitivity.ipynb`: run all cells (about 2 hours). Results are saved in `outputs/`.


