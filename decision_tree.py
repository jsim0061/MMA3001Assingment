"""Train/evaluate a decision tree with chronologically separated data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'occupancy_15min_cleaned.csv'
FEATURES = [
    'current_occupancy', 'previous_15min', 'previous_30min',
    'hour', 'minute', 'day_of_week'
]
TARGET = 'next_15min_occupancy'


def evaluate(actual, prediction, label):
    if len(actual) < 2:
        print(f'{label}: too few observations to evaluate')
        return None
    scores = {
        'split': label,
        'n': len(actual),
        'MAE': mean_absolute_error(actual, prediction),
        'RMSE': np.sqrt(mean_squared_error(actual, prediction)),
        'R2': r2_score(actual, prediction),
    }
    print(f'{label}: n={scores["n"]:,}, MAE={scores["MAE"]:.3f}, '
          f'RMSE={scores["RMSE"]:.3f}, R²={scores["R2"]:.3f}')
    return scores


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError('Run data_processing.py first to create occupancy_15min_cleaned.csv')
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET, 'collecteddate', 'target_time', 'target_is_measured']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'CSV is outdated: missing {missing}. Run data_processing.py again.')
    data['collecteddate'] = pd.to_datetime(data['collecteddate'], utc=True)
    data['target_time'] = pd.to_datetime(data['target_time'], utc=True)
    data['target_is_measured'] = data['target_is_measured'].astype('string').str.lower().eq('true').fillna(False)
    data = data.sort_values('collecteddate').reset_index(drop=True)
    if len(data) < 20:
        raise ValueError('Too few processed rows to split and train models')

    n = len(data)
    cut1, cut2 = int(0.70 * n), int(0.85 * n)
    train = data.iloc[:cut1].copy()
    validation = data.iloc[cut1:cut2].copy()
    test = data.iloc[cut2:].copy()
    # Drop examples whose future target crosses a split boundary.
    train = train.loc[train['target_time'] < validation['collecteddate'].iloc[0]]
    validation = validation.loc[validation['target_time'] < test['collecteddate'].iloc[0]]
    print('Split sizes:', len(train), len(validation), len(test))
    print(f'Zero-occupancy percentage: {100 * data["current_occupancy"].eq(0).mean():.1f}%')

    model = DecisionTreeRegressor(max_depth=20, min_samples_leaf=250, random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    predictions = {
        'Training': model.predict(train[FEATURES]),
        'Validation': model.predict(validation[FEATURES]),
        'Testing': model.predict(test[FEATURES]),
    }
    rows = []
    for label, split in [('Training', train), ('Validation', validation), ('Testing', test)]:
        row = evaluate(split[TARGET], predictions[label], label)
        if row:
            rows.append(row)

    # Independent measurement-only check: imputed zero targets aren't ground truth.
    measured = test['target_is_measured'].to_numpy(dtype=bool)
    row = evaluate(test[TARGET].to_numpy()[measured],
                   predictions['Testing'][measured], 'Testing (measured targets only)')
    if row:
        rows.append(row)
    pd.DataFrame(rows).to_csv(BASE / 'decision_tree_results.csv', index=False)

    # Plot at the target time. Select 150 consecutive test samples; gaps can still exist.
    view = test.iloc[:150]
    times = view['target_time']
    plt.figure(figsize=(12, 5))
    plt.plot(times, view[TARGET], label='Actual (processed series)')
    plt.plot(times, predictions['Testing'][:len(view)], '--', label='Tree prediction')
    plt.title('15-minute-ahead headcount: actual vs predicted')
    plt.xlabel('Time being predicted (UTC)')
    plt.ylabel('Occupants')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(BASE / 'actual_vs_predicted.png', dpi=160)
    plt.show()

    errors = test[TARGET].to_numpy() - predictions['Testing']
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30)
    plt.title('Test prediction residuals')
    plt.xlabel('Actual minus predicted (occupants)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(BASE / 'prediction_errors.png', dpi=160)
    plt.show()

    pd.Series(model.feature_importances_, index=FEATURES).sort_values().plot.barh()
    plt.title('Decision tree feature importance')
    plt.tight_layout()
    plt.savefig(BASE / 'feature_importance.png', dpi=160)
    plt.show()

    plt.figure(figsize=(16, 8))
    plot_tree(model, feature_names=FEATURES, max_depth=3, filled=True, rounded=True, fontsize=8)
    plt.title('First levels of decision tree')
    plt.tight_layout()
    plt.savefig(BASE / 'tree_structure.png', dpi=160)
    plt.show()
    print('Saved evaluation CSV and four PNG figures.')


if __name__ == '__main__':
    main()
