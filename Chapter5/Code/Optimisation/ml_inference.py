import json

import pandas as pd
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model('final_catboost_model.cbm')

with open('final_catboost_model.meta.json') as f:
    meta = json.load(f)
features = meta['feature_order']
mapping = {int(k): v for k, v in meta['target_mapping'].items()}


def predict_inspection_result(row):
    x = pd.DataFrame([[row[k] for k in features]], columns=features)
    pred = int(model.predict(x).item())
    return mapping[pred]