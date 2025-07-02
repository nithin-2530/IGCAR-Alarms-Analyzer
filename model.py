import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px
import os
# This script processes alarm data, detects anomalies using Isolation Forest,
# and visualizes results with PCA.

# 1) Load & clean

progress = {"percent": 0}  # shared progress tracker


def load_and_clean(path):
    required_cols = {'DateTime', 'Value', 'Level'}

    # Load all sheets without parse_dates
    xls = pd.read_excel(path, sheet_name=None)

    # Find the sheet with all required columns
    for sheet_name, df in xls.items():
        if required_cols.issubset(df.columns):
            # Convert DateTime column to datetime if found
            df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
            break
    else:
        raise ValueError("No sheet found with required columns: DateTime, Value, Level")

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Sort by TagName and DateTime (use DateTime only if TagName missing)
    sort_cols = ['DateTime']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Forward and backward fill Value and Level
    df[['Value', 'Level']] = df[['Value', 'Level']].ffill().bfill()

    # Drop rows with any missing values in critical columns
    required_all = required_cols.union({'TagName', 'SystemName'})
    df = df.dropna(subset=[col for col in required_all if col in df.columns])

    # Remove rows where Value is an integer (keep floats only)
    df = df[df['Value'] % 1 != 0]

    # Format DateTime as string without 'T'
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')

    return df


# 2) Encode categoricals
def encode_categoricals(df):
    le_tag = LabelEncoder()
    le_sys = LabelEncoder()
    df['Tag_enc'] = le_tag.fit_transform(df['TagName'])
    df['Sys_enc'] = le_sys.fit_transform(df['SystemName'])
    return df, le_tag, le_sys


# 3) Rolling statistics and FFT features
def compute_time_features(df, window='10s'):
    df['DateTime'] = pd.to_datetime(df['DateTime'])  # ensure datetime
    features = []

    for tag, group in df.groupby('TagName'):
        group = group.sort_values('DateTime').set_index('DateTime')

        rolled = group['Value'].rolling(window=window)

        # Rolling stats
        group['roll_mean'] = rolled.mean()
        group['roll_std']  = rolled.std()
        group['roll_min']  = rolled.min()
        group['roll_max']  = rolled.max()
        group['roll_range'] = group['roll_max'] - group['roll_min']

        # FFT spectral feature
        def spectral_max(x):
            y = np.asarray(x)
            if len(y) < 2:
                return np.nan
            mag = np.abs(fft(y))
            return mag.max()

        group['fft_max'] = rolled.apply(spectral_max, raw=False)

        # Reset index and add to result
        group = group.reset_index()
        features.append(group)

    # Combine all processed groups
    result = pd.concat(features).sort_values(['TagName', 'DateTime']).reset_index(drop=True)
    return result


# 4) Assemble X, scale
def assemble_and_scale(df, feature_cols):
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# 5) Fit IsolationForest (or classifier)
def fit_isolation_forest(X, n_estimators=100, contamination='auto', random_state=42):
    clf = IsolationForest(n_estimators=n_estimators,
                          contamination=contamination,
                          random_state=random_state,
                          n_jobs=-1)
    clf.fit(X)
    return clf

# 6) Tag outliers
def tag_outliers(df, clf, X_scaled):
    df['anomaly_score'] = clf.decision_function(X_scaled)
    # +1 normal, -1 anomaly
    df['anomaly'] = clf.predict(X_scaled)
    return df


def compute_anomaly_stats(df):
    total_rows = len(df)
    total_anomalies = (df['anomaly'] == -1).sum()
    correct_alarms = (df['anomaly'] == 1).sum()

    # All anomalous tags (with count)
    tag_counts = df[df['anomaly'] == -1]['TagName'].value_counts()

    # All anomalous systems (with count)
    sys_counts = df[df['anomaly'] == -1]['SystemName'].value_counts()

    # All tag + system anomaly pairs (with count)
    pair_counts = (
        df[df['anomaly'] == -1]
        .groupby(['TagName', 'SystemName'])
        .size()
        .sort_values(ascending=False)
    )

    return {
        'total_rows': total_rows,
        'total_anomalies': total_anomalies,
        'correct_alarms': correct_alarms,
        'all_tags': tag_counts.to_dict(),
        'all_systems': sys_counts.to_dict(),
        'all_pairs': [{"tag": tag, "system": system, "count": count} for (tag, system), count in pair_counts.items()]
    }



def visualize_pca_plotly(X_scaled, df):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    df = df.copy()
    df['PCA1'] = components[:, 0]
    df['PCA2'] = components[:, 1]
    df['AnomalyLabel'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    fig = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='AnomalyLabel',
        hover_data=['DateTime', 'TagName', 'SystemName','Value','Level'] if all(col in df.columns for col in ['DateTime', 'TagName', 'SystemName','Value','Level']) else None,
        title='Alarm Distribution View',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        labels={'PCA1': 'Pattern Axis 1', 'PCA2': 'Pattern Axis 2'},
        render_mode='webgl'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=600)

    return fig.to_html(full_html=False)

def process_excel_pipeline(file_path, output_dir):
    progress["percent"] = 0

    df = load_and_clean(file_path)
    progress["percent"] = 20

    df, le_tag, le_sys = encode_categoricals(df)
    progress["percent"] = 40

    df = compute_time_features(df, window='10s')
    progress["percent"] = 60

    feature_cols = [
        'Tag_enc', 'Sys_enc', 'Level',
        'roll_mean', 'roll_std', 'roll_min', 'roll_max', 'roll_range',
        'fft_max'
    ]
    X_scaled, scaler = assemble_and_scale(df, feature_cols)
    progress["percent"] = 75

    clf = fit_isolation_forest(X_scaled, n_estimators=100, contamination=0.01, random_state=42)
    df = tag_outliers(df, clf, X_scaled)
    progress["percent"] = 90

    output_filename = f"anomaly_{os.path.basename(file_path)}"
    output_path = os.path.join(output_dir, output_filename)
    df.to_excel(output_path, index=False)
    progress["percent"] = 100

    plot_html = visualize_pca_plotly(X_scaled, df)

    stats = compute_anomaly_stats(df)

    return plot_html, output_filename, stats

