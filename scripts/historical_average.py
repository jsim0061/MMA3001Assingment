"""Weekday and time-of-day historical-mean occupancy baseline."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from _common import ROOT, TARGET, load_data, chronological_split, save_evaluations

def clock_columns(part):
    """Extract Melbourne-local target weekday, hour and 15-minute clock slot."""
    clock=part['target_time'].dt.tz_convert('Australia/Melbourne')
    return pd.DataFrame({'dow':clock.dt.dayofweek.to_numpy(),
      'hour':clock.dt.hour.to_numpy(),'minute':clock.dt.minute.to_numpy()})

def main():
    """Learn training-only calendar averages and score later observations."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'occupancy_15min_cleaned.csv')
    args=ap.parse_args(); splits=chronological_split(load_data(args.data));train=splits[0]
    history=clock_columns(train);history[TARGET]=train[TARGET].to_numpy()
    means=history.groupby(['dow','hour','minute'])[TARGET].mean()
    by_hour=history.groupby('hour')[TARGET].mean();global_mean=float(train[TARGET].mean())
    def predict(part):
        """Look up training averages using the future target's local time."""
        clock=clock_columns(part)
        keys=pd.MultiIndex.from_frame(clock[['dow','hour','minute']])
        out=means.reindex(keys).to_numpy(dtype=float)
        hourly=clock['hour'].map(by_hour).to_numpy(dtype=float)
        out=np.where(np.isnan(out),hourly,out)
        return np.where(np.isnan(out),global_mean,out)
    save_evaluations('historical_average',None,splits,predict,ROOT/'results')
if __name__=='__main__':main()
