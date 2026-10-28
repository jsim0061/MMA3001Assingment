"""Prepare one occupancy space as a 15-minute prediction time series."""
from pathlib import Path
import zipfile
import pandas as pd

BASE = Path(__file__).resolve().parent
ZIP_PATH = BASE / '5occupancySensor_MayToDec2024_9MRows.zip'
OUTPUT_PATH = BASE / 'occupancy_15min_cleaned.csv'
ROOM_ID = '0f3f37f8-0335-41e0-ae87-a1d677dfc7d7'
CHUNK_SIZE = 100_000
MAX_GAP = pd.Timedelta(minutes=30)


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f'Place the ZIP next to this script: {ZIP_PATH}')
    parts = []
    total = 0
    with zipfile.ZipFile(ZIP_PATH) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if not csv_names:
            raise ValueError('ZIP does not contain a CSV file')
        with archive.open(csv_names[0]) as file:
            for chunk in pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                usecols=['floorspaceid', 'collecteddate', 'headcount'],
                dtype={'floorspaceid': 'string'},
            ):
                total += len(chunk)
                chosen = chunk.loc[chunk['floorspaceid'].eq(ROOM_ID),
                                   ['collecteddate', 'headcount']].copy()
                if len(chosen):
                    parts.append(chosen)
                print(f'Read {total:,} rows; matched {sum(len(p) for p in parts):,}', flush=True)
    if not parts:
        raise ValueError(f'No records for ROOM_ID {ROOM_ID}. Check the selected floorspaceid.')
    df = pd.concat(parts, ignore_index=True)
    df['collecteddate'] = pd.to_datetime(df['collecteddate'], errors='coerce', utc=True)
    df['headcount'] = pd.to_numeric(df['headcount'], errors='coerce')
    df = df.dropna(subset=['collecteddate', 'headcount'])
    df = df.loc[(df['headcount'] >= 0) & (df['headcount'] % 1 == 0)]
    df = df.sort_values('collecteddate', kind='stable')
    conflicts = (df.groupby('collecteddate')['headcount'].nunique() > 1).sum()
    print(f'Timestamps with conflicting headcounts: {conflicts:,}')
    # Consistent, documented rule for duplicate timestamps.
    df = df.drop_duplicates(subset='collecteddate', keep='last')
    df = df.set_index('collecteddate')

    # Every bin is labelled by its ending time (UTC avoids DST ambiguities).
    grouped = df.resample('15min', label='right', closed='right')
    data = grouped['headcount'].last().to_frame('measured_occupancy')
    # Last real measurement time in the bin, not the bin end.
    source_times = pd.Series(df.index, index=df.index, name='timestamp')
    data['measurement_time'] = source_times.resample('15min', label='right', closed='right').last()
    data['is_measured'] = data['measured_occupancy'].notna()
    data['last_measurement'] = data['measurement_time'].ffill()
    data['minutes_since_measurement'] = (
        (data.index.to_series() - data['last_measurement']).dt.total_seconds() / 60
    )
    data['assumed_empty'] = data['last_measurement'].notna() & (
        data['minutes_since_measurement'] > MAX_GAP.total_seconds() / 60
    )
    data['current_occupancy'] = data['measured_occupancy'].ffill()
    data.loc[data['assumed_empty'], 'current_occupancy'] = 0
    # Real measurements always override an assumed zero.
    data.loc[data['is_measured'], 'current_occupancy'] = data.loc[
        data['is_measured'], 'measured_occupancy'
    ]
    data['previous_15min'] = data['current_occupancy'].shift(1)
    data['previous_30min'] = data['current_occupancy'].shift(2)
    data['next_15min_occupancy'] = data['current_occupancy'].shift(-1)
    data['target_time'] = data.index + pd.Timedelta(minutes=15)
    data['target_is_measured'] = data['is_measured'].shift(-1).fillna(False).astype(bool)
    local = data.index.tz_convert('Australia/Melbourne')
    data['hour'] = local.hour
    data['minute'] = local.minute
    data['day_of_week'] = local.dayofweek
    data = data.dropna(subset=[
        'current_occupancy', 'previous_15min', 'previous_30min', 'next_15min_occupancy'
    ]).reset_index()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved {len(data):,} rows to {OUTPUT_PATH}')
    print(f'Assumed empty rows: {int(data["assumed_empty"].sum()):,}')
    print(f'Measured targets: {int(data["target_is_measured"].sum()):,}')
    print(data.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
