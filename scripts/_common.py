"""Shared command-line script helpers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from occupancy_prediction.core import FEATURES, TARGET, load_data, chronological_split, scores
import pandas as pd

def save_evaluations(name, model, splits, predict, folder):
    """Save train/validation/test scores, measured-only test scores, and test predictions."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,part in zip(['Training','Validation','Testing'],splits):
        prediction=predict(part)
        rows.append({'Dataset':label,**scores(part[TARGET],prediction)})
        if label=='Testing':
            output=pd.DataFrame({'target_time':part['target_time'].to_numpy(),
                   'actual_occupancy':part[TARGET].to_numpy(), 'predicted_occupancy':prediction})
            if 'target_is_measured' in part:
                output['target_is_measured']=part['target_is_measured'].to_numpy()
                mask=part['target_is_measured'].to_numpy(dtype=bool)
                if mask.sum()>=2:
                    rows.append({'Dataset':'Measured Targets Only',**scores(part[TARGET].to_numpy()[mask],prediction[mask])})
            output.to_csv(folder/f'{name}_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/f'{name}_results.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
