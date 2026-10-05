# ML training

Trains the CatBoost classifier used in the agent-based model (ABM) to predict
the compliance change of food business operators (FBOs) between inspections
(-1 = improvement, 0 = no change, +1 = deterioration)

## Files

| File | Description |
|---|---|
| `1_train_ml_model.ipynb` | Training notebook |
| `ML_train.xlsx` | Training data|
| `final_catboost_model.cbm` | Trained model |
| `final_catboost_model.meta.json` | Feature order and class mapping used by the ABM |
| `requirements.txt` | Required Python packages |

## How to run

1. Install the required packages: `pip install -r requirements.txt`
2. Open `1_train_ml_model.ipynb` in JupyterLab and run all cells.

Running the notebook saves a new model as `retrained_catboost_model.cbm`
(and `retrained_catboost_model.meta.json`), so the model used in the thesis is not overwritten.
