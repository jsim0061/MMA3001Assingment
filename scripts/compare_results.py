"""Combine test-set metric CSVs produced by the four scripts."""
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
items=[('Decision Tree',root/'decision_tree_results.csv'),('Historical Average',root/'results/historical_average_results.csv'),('Linear Regression',root/'results/linear_regression_results.csv'),('SVR',root/'results/svr_results.csv')]
rows=[]
for label,file in items:
    if not file.exists():
        print(f'Skipping {label}: missing {file}')
        continue
    table=pd.read_csv(file);col='Dataset' if 'Dataset' in table.columns else 'split'
    subset=table[table[col].astype(str).str.lower().eq('testing')]
    if not subset.empty:
        score=subset.iloc[0]
        rows.append({'Model':label,'MAE':float(score['MAE']),'RMSE':float(score['RMSE']),'R2':float(score['R2'])})
output=pd.DataFrame(rows)
(root/'results').mkdir(exist_ok=True)
output.to_csv(root/'results/model_comparison.csv',index=False)
print(output.to_string(index=False) if len(output) else 'No model results found')
