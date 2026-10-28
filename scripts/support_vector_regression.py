"""Sampled RBF-SVR comparator, selected using validation MAE only."""
import argparse
from pathlib import Path
import pandas as pd
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from _common import ROOT, FEATURES, TARGET, load_data, chronological_split, save_evaluations

def main():
    """Tune sampled SVR on validation data and report held-out testing scores."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'occupancy_15min_cleaned.csv')
    ap.add_argument('--max-train-samples',type=int,default=5000)
    args=ap.parse_args();splits=chronological_split(load_data(args.data));train,val,test=splits
    sample=train.sample(min(args.max_train_samples,len(train)),random_state=42)
    trials=[];best=None;best_mae=float('inf')
    for c,eps,gamma in product([1,10,100],[.1,.5],['scale',.1]):
        model=make_pipeline(StandardScaler(),SVR(kernel='rbf',C=c,epsilon=eps,gamma=gamma))
        model.fit(sample[FEATURES],sample[TARGET]);mae=mean_absolute_error(val[TARGET],model.predict(val[FEATURES]))
        trials.append({'C':c,'epsilon':eps,'gamma':gamma,'validation_MAE':mae})
        if mae<best_mae:best,best_mae=model,mae
        print(f'C={c}, eps={eps}, gamma={gamma}: validation MAE={mae:.3f}',flush=True)
    (ROOT/'results').mkdir(exist_ok=True)
    pd.DataFrame(trials).to_csv(ROOT/'results'/'svr_parameter_results.csv',index=False)
    # Training score is explicitly on sampled rows, unlike the other methods.
    save_evaluations('svr',best,(sample,val,test),lambda p:best.predict(p[FEATURES]),ROOT/'results')
if __name__=='__main__':main()
