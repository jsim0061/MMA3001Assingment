"""Reusable, testable operations for 15-minute-ahead room occupancy prediction.

All time-indexed examples must be sorted before splitting. Raw timestamps are
UTC; calendar features are derived in Australia/Melbourne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ('current_occupancy', 'previous_15min', 'previous_30min', 'hour', 'minute', 'day_of_week')
TARGET = 'next_15min_occupancy'


def validate_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort processed occupancy examples.

    Args:
        data: DataFrame with six numeric features, prediction target,
            ``collecteddate`` and ``target_time`` timestamps.

    Returns:
        Chronologically sorted copy with UTC datetime columns.

    Raises:
        ValueError: Required columns missing, invalid numbers, invalid times,
            non-15-minute prediction horizon, or too few rows.
    """
    required = list(FEATURES) + [TARGET, 'collecteddate', 'target_time']
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = data.copy()
    for name in FEATURES + (TARGET,):
        df[name] = pd.to_numeric(df[name], errors='coerce')
    if not np.isfinite(df[list(FEATURES) + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError('Inputs and target must contain finite numeric values')
    if (df[list(FEATURES[:3]) + [TARGET]] < 0).any().any():
        raise ValueError('Occupancy headcounts cannot be negative')
    df['collecteddate'] = pd.to_datetime(df['collecteddate'], utc=True, errors='coerce')
    df['target_time'] = pd.to_datetime(df['target_time'], utc=True, errors='coerce')
    if df[['collecteddate', 'target_time']].isna().any().any():
        raise ValueError('Invalid timestamps')
    if not ((df['target_time'] - df['collecteddate']) == pd.Timedelta(minutes=15)).all():
        raise ValueError('Every prediction target must be exactly 15 minutes ahead')
    if not df['hour'].between(0, 23).all() or not df['minute'].isin([0,15,30,45]).all() or not df['day_of_week'].between(0,6).all():
        raise ValueError('Time features outside expected ranges')
    if len(df) < 20:
        raise ValueError('At least 20 examples required')
    return df.sort_values('collecteddate').reset_index(drop=True)


def chronological_split(data: pd.DataFrame, train_fraction: float = .70, validation_fraction: float = .15):
    """Split sorted time series into train/validation/test without crossing target boundaries.

    Args:
        data: Frame accepted by :func:`validate_frame`.
        train_fraction: Fraction of earlier examples assigned to training.
        validation_fraction: Fraction assigned to validation.

    Returns:
        Tuple of (train, validation, test) DataFrames.

    Raises:
        ValueError: Fractions are invalid or any resulting split is empty.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError('Invalid split fractions')
    df = validate_frame(data)
    n = len(df)
    a, b = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    train, validation, test = df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
    if validation.empty or test.empty:
        raise ValueError('Not enough rows for three nonempty splits')
    train = train[train.target_time < validation.collecteddate.iloc[0]].copy()
    validation = validation[validation.target_time < test.collecteddate.iloc[0]].copy()
    if train.empty or validation.empty:
        raise ValueError('Boundary filtering emptied a split')
    return train, validation, test


def evaluate_regression(actual, predicted) -> dict[str, float]:
    """Compute MAE, RMSE (people) and R2 (dimensionless).

    Args:
        actual: Recorded target occupancy values.
        predicted: Corresponding model predictions.

    Returns:
        Dictionary containing MAE, RMSE, and R2.

    Raises:
        ValueError: Fewer than two observations or invalid/mismatched arrays.
    """
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) < 2 or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Need two or more paired finite 1D predictions')
    return {'MAE': float(mean_absolute_error(y, p)),
            'RMSE': float(np.sqrt(mean_squared_error(y, p))),
            'R2': float(r2_score(y, p))}


def measured_target_mask(data: pd.DataFrame) -> np.ndarray:
    """Return boolean mask of genuinely measured target observations.

    Args:
        data: Frame containing ``target_is_measured``.

    Returns:
        Boolean NumPy array of length ``len(data)``.

    Raises:
        ValueError: Measurement flag column is missing.
    """
    if 'target_is_measured' not in data:
        raise ValueError('target_is_measured is required')
    return data['target_is_measured'].astype('string').str.lower().eq('true').fillna(False).to_numpy(dtype=bool)


def historical_average_fit(train: pd.DataFrame) -> dict:
    """Fit time-of-week occupancy averages using training targets only.

    Args:
        train: Chronological training frame with target timestamps and headcount.

    Returns:
        Learned day/quarter-hour means, hour means and global mean.
    """
    frame = _historical_keys(train)
    return {
        'slot': frame.groupby(['weekday', 'hour', 'minute'])[TARGET].mean(),
        'hour': frame.groupby('hour')[TARGET].mean(),
        'global': float(frame[TARGET].mean()),
    }


def _historical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add target-time-of-week calendar fields using Melbourne local time."""
    out = frame.copy()
    t = pd.to_datetime(out['target_time'], utc=True).dt.tz_convert('Australia/Melbourne')
    out['weekday'], out['hour'], out['minute'] = t.dt.dayofweek, t.dt.hour, t.dt.minute
    return out


def historical_average_predict(fit: dict, data: pd.DataFrame) -> np.ndarray:
    """Predict using trained weekly slot means, then hour and global fallbacks.

    Args:
        fit: Mapping returned by :func:`historical_average_fit`.
        data: DataFrame with UTC ``target_time`` values.

    Returns:
        Float array of predicted occupants, one per row.
    """
    df = _historical_keys(data)
    ix = pd.MultiIndex.from_frame(df[['weekday', 'hour', 'minute']])
    vals = fit['slot'].reindex(ix).to_numpy(dtype=float)
    fallback = df['hour'].map(fit['hour']).to_numpy(dtype=float)
    vals = np.where(np.isnan(vals), fallback, vals)
    return np.where(np.isnan(vals), fit['global'], vals)
