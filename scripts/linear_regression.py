"""Fit an untuned linear regression comparator on the fixed room time split."""
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
from _common import ROOT, FEATURES, TARGET, load_data, chronological_split, save_evaluations

def main():
    """Train LinearRegression on past data and save all evaluation metrics."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'occupancy_15min_cleaned.csv')
    args=ap.parse_args()
    splits=chronological_split(load_data(args.data))
    model=LinearRegression().fit(splits[0][FEATURES],splits[0][TARGET])
    save_evaluations('linear_regression',model,splits,lambda p:model.predict(p[FEATURES]),ROOT/'results')
if __name__=='__main__':main()
