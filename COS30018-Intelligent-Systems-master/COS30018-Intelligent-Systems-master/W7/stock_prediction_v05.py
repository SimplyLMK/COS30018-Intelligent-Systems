# File: stock_prediction_v05.py
# Authors: Le Minh Kha (based on work by Bao Vo, Cheong Koo, and x4nth055)
# Version: 0.5 - Machine Learning 2 (Multivariate & Multistep Prediction)
#
# This version builds on v0.4 by adding:
# - Multistep prediction: predict k future closing prices simultaneously
# - Explicit multivariate experiments: compare univariate vs multivariate input
# - Combined multivariate + multistep prediction
# - evaluate_multistep() function for per-step and overall metrics
# - Visualization: trajectory plots and per-step error charts
# All previous functions (load_data, create_model, visualizations) are preserved.

#------------------------------------------------------------------------------
# IMPORT LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import time
import yfinance as yf
import tensorflow as tf
import random
import mplfinance as mpf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional
from collections import deque

#------------------------------------------------------------------------------
# SET RANDOM SEEDS FOR REPRODUCIBILITY
#------------------------------------------------------------------------------
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

#------------------------------------------------------------------------------
# LAYER TYPE MAPPING
#------------------------------------------------------------------------------
# Maps string names to Keras layer classes, used by create_model() to
# dynamically instantiate layers from configuration dicts.
LAYER_MAP = {
    "LSTM": tf.keras.layers.LSTM,
    "GRU": tf.keras.layers.GRU,
    "SimpleRNN": tf.keras.layers.SimpleRNN,
    "Dense": tf.keras.layers.Dense
}

# Set of recurrent layer types (need return_sequences logic and dropout)
RECURRENT_TYPES = {"LSTM", "GRU", "SimpleRNN"}


#------------------------------------------------------------------------------
# DYNAMIC MODEL BUILDER (from v0.4, updated for multistep output)
#------------------------------------------------------------------------------
def create_model(input_shape, layer_configs, dropout_rate=0.2,
                 loss="mean_absolute_error", optimizer="adam",
                 output_steps=1):
    """
    Build a Deep Learning model dynamically from a configuration list.

    Instead of hardcoding layers, this function reads a list of dicts
    describing each layer and builds the model accordingly. This allows
    easy experimentation with different architectures.

    Parameters:
        input_shape: tuple, shape of input data (n_steps, n_features)
        layer_configs: list of dicts, each with:
            - "type": str — one of "LSTM", "GRU", "SimpleRNN", "Dense"
            - "units": int — number of units/neurons
            - "params": dict (optional) — extra kwargs passed to the layer
        dropout_rate: float, dropout applied after each recurrent layer
                      (0 = no dropout). Not applied after Dense layers.
        loss: str, loss function name
        optimizer: str, optimizer name
        output_steps: int, number of output values (1 for single-step,
                      k for k-step-ahead multistep prediction)

    Returns:
        compiled tf.keras.Model

    Raises:
        ValueError: if layer_configs is empty or contains unknown layer types

    Example:
        model = create_model(
            input_shape=(50, 5),
            layer_configs=[
                {"type": "LSTM", "units": 50},
                {"type": "LSTM", "units": 50},
            ],
            dropout_rate=0.2,
            output_steps=5  # predict 5 days ahead
        )
    """
    # --- Validation ---
    if not layer_configs:
        raise ValueError("layer_configs cannot be empty.")

    for i, cfg in enumerate(layer_configs):
        if cfg["type"] not in LAYER_MAP:
            raise ValueError(
                f"Unknown layer type '{cfg['type']}' at index {i}. "
                f"Valid types: {list(LAYER_MAP.keys())}")

    # --- Auto-append Dense(output_steps) if last layer is not Dense ---
    # Updated from v0.4: uses output_steps instead of hardcoded 1
    if layer_configs[-1]["type"] != "Dense":
        layer_configs = layer_configs + [{"type": "Dense", "units": output_steps}]

    # --- Determine return_sequences for each recurrent layer ---
    # A recurrent layer needs return_sequences=True if any SUBSEQUENT layer
    # in the config is also recurrent (i.e., stacked recurrent layers).
    # The last recurrent layer before a Dense layer returns a 2D tensor.
    def needs_return_sequences(index):
        """Check if any layer after 'index' is a recurrent type."""
        for j in range(index + 1, len(layer_configs)):
            if layer_configs[j]["type"] in RECURRENT_TYPES:
                return True
        return False

    # --- Build the Sequential model ---
    model = Sequential()

    for i, cfg in enumerate(layer_configs):
        layer_class = LAYER_MAP[cfg["type"]]
        units = cfg["units"]
        extra_params = cfg.get("params", {})  # Optional extra kwargs

        if cfg["type"] in RECURRENT_TYPES:
            # Set return_sequences based on whether more recurrent layers follow
            return_seq = needs_return_sequences(i)

            if i == 0:
                # First layer gets input_shape so the model knows input dimensions
                model.add(layer_class(
                    units,
                    return_sequences=return_seq,
                    input_shape=input_shape,
                    **extra_params
                ))
            else:
                model.add(layer_class(
                    units,
                    return_sequences=return_seq,
                    **extra_params
                ))

            # Add dropout after each recurrent layer (skip if rate is 0)
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        else:
            # Dense layer (no return_sequences, no dropout)
            if i == 0:
                model.add(layer_class(units, input_shape=input_shape,
                                      **extra_params))
            else:
                model.add(layer_class(units, **extra_params))

    # --- Compile the model ---
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


#------------------------------------------------------------------------------
# HELPER FUNCTION: SHUFFLE TWO ARRAYS IN UNISON
#------------------------------------------------------------------------------
def shuffle_in_unison(a, b):
    """
    Shuffle two arrays in the same way (maintaining correspondence).
    Captures the RNG state before shuffling 'a', resets it, then shuffles 'b'
    so both arrays undergo the identical permutation.

    Parameters:
        a (np.array): First array (typically X - features)
        b (np.array): Second array (typically y - labels)
    """
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


#------------------------------------------------------------------------------
# MAIN FUNCTION: LOAD AND PROCESS DATA (updated for multistep targets)
#------------------------------------------------------------------------------
def load_data(ticker,
              start_date=None,
              end_date=None,
              n_steps=50,
              scale=True,
              shuffle=True,
              lookup_step=1,
              n_future_steps=1,
              split_by_date=True,
              test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              save_local=False,
              load_local=False,
              local_path='data'):
    """
    Load stock data from Yahoo Finance and process it for LSTM model training.

    Updated in v0.5: Added n_future_steps parameter for multistep prediction.

    Parameters:
        ticker: str or pd.DataFrame — stock ticker or pre-loaded DataFrame
        start_date: str — start date for data download ('YYYY-MM-DD')
        end_date: str — end date for data download ('YYYY-MM-DD')
        n_steps: int — number of lookback days (sequence length)
        scale: bool — whether to scale features to [0, 1]
        shuffle: bool — whether to shuffle training/test data
        lookup_step: int — days ahead to predict (only used when n_future_steps=1)
        n_future_steps: int — number of future steps to predict (default=1).
            When > 1, targets become a vector [day+1, day+2, ..., day+k].
            The lookup_step parameter is ignored when n_future_steps > 1.
        split_by_date: bool — if True, split chronologically; if False, random split
        test_size: float — fraction of data for testing
        feature_columns: list of str — which columns to use as input features
        save_local: bool — whether to save downloaded data to CSV
        load_local: bool — whether to load from local CSV if available
        local_path: str — directory for local CSV files

    Returns:
        dict with keys: df, X_train, y_train, X_test, y_test, column_scaler,
                        test_df, last_sequence, n_future_steps
    """

    # STEP 1: DATA ACQUISITION
    if isinstance(ticker, str):
        local_file = os.path.join(local_path, f"{ticker}_{start_date}_{end_date}.csv")

        if load_local and os.path.exists(local_file):
            print(f"Loading data from local file: {local_file}")
            df = pd.read_csv(local_file, index_col=0, parse_dates=True)
            df.index.name = 'date'
        else:
            print(f"Downloading data for {ticker} from Yahoo Finance...")

            if start_date and end_date:
                df = yf.download(ticker, start=start_date, end=end_date,
                               progress=False, auto_adjust=False)
            elif start_date:
                df = yf.download(ticker, start=start_date,
                               progress=False, auto_adjust=False)
            elif end_date:
                df = yf.download(ticker, end=end_date,
                               progress=False, auto_adjust=False)
            else:
                df = yf.download(ticker, period="max",
                               progress=False, auto_adjust=False)

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize column names to lowercase with no spaces
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower().replace(' ', '')
                column_mapping[col] = col_lower
            df.rename(columns=column_mapping, inplace=True)

            # Handle missing 'adjclose' column
            if 'adjclose' not in df.columns and 'close' in df.columns:
                df['adjclose'] = df['close']

            df.index.name = 'date'

            # Save to local storage
            if save_local:
                os.makedirs(local_path, exist_ok=True)
                df.to_csv(local_file)
                print(f"Data saved to: {local_file}")

    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a pd.DataFrame instance")

    # STEP 2: NaN HANDLING
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the data")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        print(f"NaN values have been handled using forward/backward fill")

    # Initialize result dictionary
    result = {}
    result['df'] = df.copy()
    result['n_future_steps'] = n_future_steps

    # STEP 3: Validate feature columns
    for col in feature_columns:
        assert col in df.columns, \
            f"'{col}' does not exist in the dataframe. Available columns: {list(df.columns)}"

    # STEP 4: Add date column
    if "date" not in df.columns:
        df["date"] = df.index

    # STEP 5: Feature scaling
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
                np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    # STEP 6: Create target column(s)
    # NEW IN v0.5: Support for multistep targets
    if n_future_steps > 1:
        # Multistep: create k shifted columns (future_1, future_2, ..., future_k)
        for step in range(1, n_future_steps + 1):
            df[f'future_{step}'] = df['adjclose'].shift(-step)
        # Drop rows where ANY future column is NaN (last k rows)
        df.dropna(inplace=True)
    else:
        # Single-step: same as v0.4
        df['future'] = df['adjclose'].shift(-lookup_step)

    # STEP 7: Save last sequence
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # STEP 8: Drop NaN (for single-step; multistep already dropped above)
    if n_future_steps == 1:
        df.dropna(inplace=True)

    # STEP 9: Create sequences
    # NEW IN v0.5: Modified loop to handle multistep targets
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    if n_future_steps > 1:
        # Multistep: zip with a 2D targets array
        future_cols = [f'future_{s}' for s in range(1, n_future_steps + 1)]
        targets_array = df[future_cols].values  # shape: (n_rows, k)
        entries = df[feature_columns + ["date"]].values

        for i in range(len(entries)):
            sequences.append(entries[i])
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), targets_array[i]])
    else:
        # Single-step: same as v0.4
        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

    # STEP 10: Prepare last sequence
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    # STEP 11: Separate features and labels
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)

    # STEP 12: Train/test split
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]

        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = \
            train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # STEP 13: Extract test dates
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

    # STEP 14: Remove date column and convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result


#------------------------------------------------------------------------------
# NEW IN v0.5: MULTISTEP EVALUATION FUNCTION
#------------------------------------------------------------------------------
def evaluate_multistep(y_true, y_pred, scaler):
    """
    Evaluate multistep predictions with per-step and overall metrics.

    Both y_true and y_pred are in scaled [0,1] space. This function
    inverse-transforms them back to original price scale before computing
    metrics.

    Parameters:
        y_true: np.array, shape (n_samples, k) — actual future prices (scaled)
        y_pred: np.array, shape (n_samples, k) — predicted future prices (scaled)
        scaler: sklearn MinMaxScaler — the scaler for 'adjclose' column

    Returns:
        dict with:
            'per_step': list of dicts, one per step with MAE, RMSE, MAPE, R²
            'overall': dict with overall MAE, RMSE, MAPE, R²
    """
    n_steps_ahead = y_true.shape[1]
    per_step_results = []

    # Per-step metrics: inverse transform each column independently
    for step in range(n_steps_ahead):
        true_col = scaler.inverse_transform(y_true[:, step].reshape(-1, 1)).flatten()
        pred_col = scaler.inverse_transform(y_pred[:, step].reshape(-1, 1)).flatten()

        mae = mean_absolute_error(true_col, pred_col)
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        mape = np.mean(np.abs((true_col - pred_col) / true_col)) * 100
        r2 = r2_score(true_col, pred_col)

        per_step_results.append({
            'step': step + 1,
            'MAE ($)': round(mae, 2),
            'RMSE ($)': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'R²': round(r2, 4)
        })

    # Overall metrics: flatten all steps together
    y_true_flat = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_flat = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    overall = {
        'MAE ($)': round(mean_absolute_error(y_true_flat, y_pred_flat), 2),
        'RMSE ($)': round(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)), 2),
        'MAPE (%)': round(np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100, 2),
        'R²': round(r2_score(y_true_flat, y_pred_flat), 4)
    }

    return {'per_step': per_step_results, 'overall': overall}


#------------------------------------------------------------------------------
# NEW IN v0.5: MULTISTEP TRAJECTORY PLOT
#------------------------------------------------------------------------------
def plot_multistep_trajectory(y_true, y_pred, scaler, n_trajectories=5,
                              title="Multistep Prediction Trajectories",
                              save_path=None):
    """
    Plot predicted vs actual k-day trajectories for a sample of test points.

    Selects n_trajectories evenly spaced test samples and overlays the
    predicted trajectory against the actual future prices.

    Parameters:
        y_true: np.array, shape (n_samples, k) — actual future prices (scaled)
        y_pred: np.array, shape (n_samples, k) — predicted future prices (scaled)
        scaler: sklearn MinMaxScaler — the scaler for 'adjclose' column
        n_trajectories: int — number of sample trajectories to plot
        title: str — chart title
        save_path: str or None — if provided, saves the figure to this path
    """
    n_samples = y_true.shape[0]
    n_steps_ahead = y_true.shape[1]

    # Select evenly spaced indices
    indices = np.linspace(0, n_samples - 1, n_trajectories, dtype=int)

    fig, axes = plt.subplots(1, n_trajectories, figsize=(4 * n_trajectories, 4),
                              sharey=True)
    if n_trajectories == 1:
        axes = [axes]

    steps = list(range(1, n_steps_ahead + 1))

    for idx, (ax, sample_idx) in enumerate(zip(axes, indices)):
        true_vals = scaler.inverse_transform(
            y_true[sample_idx].reshape(-1, 1)).flatten()
        pred_vals = scaler.inverse_transform(
            y_pred[sample_idx].reshape(-1, 1)).flatten()

        ax.plot(steps, true_vals, 'b-o', label='Actual', markersize=4)
        ax.plot(steps, pred_vals, 'r--s', label='Predicted', markersize=4)
        ax.set_title(f'Sample {sample_idx}', fontsize=10)
        ax.set_xlabel('Steps Ahead')
        if idx == 0:
            ax.set_ylabel('Price ($)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")
    plt.show()


#------------------------------------------------------------------------------
# NEW IN v0.5: PER-STEP ERROR BAR CHART
#------------------------------------------------------------------------------
def plot_per_step_error(per_step_results, metric='MAPE (%)',
                        title="Per-Step Prediction Error",
                        save_path=None):
    """
    Bar chart showing how prediction error changes with forecast horizon.

    Parameters:
        per_step_results: list of dicts from evaluate_multistep()['per_step']
        metric: str — which metric to plot ('MAE ($)', 'RMSE ($)', 'MAPE (%)', 'R²')
        title: str — chart title
        save_path: str or None — if provided, saves the figure to this path
    """
    steps = [r['step'] for r in per_step_results]
    values = [r[metric] for r in per_step_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(steps, values, color='steelblue', edgecolor='black', alpha=0.8)

    # Add value labels on each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Steps Ahead', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(steps)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-step error plot saved to: {save_path}")
    plt.show()


#==============================================================================
# FROM v0.3: VISUALIZATION FUNCTIONS (Task C.3)
#==============================================================================

#------------------------------------------------------------------------------
# TASK C.3.1: CANDLESTICK CHART
#------------------------------------------------------------------------------
def plot_candlestick(df, n_days=1, ticker="Stock", start_date=None, end_date=None):
    """
    Display stock market financial data using a candlestick chart.
    (See v0.3/v0.4 for full documentation)
    """
    data = df.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data.index.name = 'Date'

    cols_lower = {col: col.lower() for col in data.columns}
    ohlcv = {}

    for col, col_lower in cols_lower.items():
        if col_lower == 'open':
            ohlcv['Open'] = data[col]
        elif col_lower == 'high':
            ohlcv['High'] = data[col]
        elif col_lower == 'low':
            ohlcv['Low'] = data[col]
        elif col_lower == 'volume':
            ohlcv['Volume'] = data[col]

    if 'adjclose' in cols_lower.values():
        adjclose_col = [c for c, cl in cols_lower.items() if cl == 'adjclose'][0]
        ohlcv['Close'] = data[adjclose_col]
    elif 'close' in cols_lower.values():
        close_col = [c for c, cl in cols_lower.items() if cl == 'close'][0]
        ohlcv['Close'] = data[close_col]

    data = pd.DataFrame(ohlcv, index=data.index)

    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]

    if n_days > 1:
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
        }
        if 'Volume' in data.columns:
            ohlc_dict['Volume'] = 'sum'
        data = data.resample(f'{n_days}D').agg(ohlc_dict).dropna()

    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {list(data.columns)}. "
                f"Ensure the DataFrame has OHLC data.")

    market_colors = mpf.make_marketcolors(
        up='green', down='red', wick='inherit', edge='inherit', volume='in')
    chart_style = mpf.make_mpf_style(
        marketcolors=market_colors, gridstyle='--', gridaxis='both')

    if n_days == 1:
        title_str = f"{ticker} Candlestick Chart (Daily)"
    else:
        title_str = f"{ticker} Candlestick Chart ({n_days}-Day Candles)"

    show_volume = 'Volume' in data.columns

    mpf.plot(
        data, type='candle', style=chart_style,
        title=title_str, ylabel='Price ($)',
        volume=show_volume, ylabel_lower='Volume',
        figsize=(14, 8), tight_layout=True
    )


#------------------------------------------------------------------------------
# TASK C.3.2: BOXPLOT CHART
#------------------------------------------------------------------------------
def plot_boxplot(df, n_days=5, column='adjclose', ticker="Stock",
                 start_date=None, end_date=None):
    """
    Display stock market financial data using a boxplot chart.
    (See v0.3/v0.4 for full documentation)
    """
    data = df.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]

    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found. Available: {list(data.columns)}")

    window_data = []
    window_labels = []

    for i in range(0, len(data), n_days):
        window = data.iloc[i:i + n_days]
        if len(window) < n_days:
            break
        window_data.append(window[column].values)
        label_start = window.index[0].strftime('%Y-%m-%d')
        label_end = window.index[-1].strftime('%Y-%m-%d')
        window_labels.append(f"{label_start}\nto\n{label_end}")

    if len(window_data) == 0:
        print("Warning: Not enough data to create boxplots with the given "
              f"window size of {n_days} days.")
        return

    fig_width = max(10, min(len(window_data) * 0.8, 30))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bp = ax.boxplot(
        window_data, labels=window_labels, patch_artist=True,
        boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.2),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.0),
        capprops=dict(color='black', linewidth=1.0),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.6),
        widths=0.6
    )

    ax.set_title(
        f"{ticker} {column.title()} - Boxplot ({n_days}-Day Moving Window)",
        fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{column.title()} ($)', fontsize=12)
    ax.set_xlabel('Trading Period', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


#==============================================================================
# MAIN EXECUTION
#==============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("STOCK PREDICTION v0.5 - Machine Learning 2")
    print("Multivariate & Multistep Prediction")
    print("=" * 60)

    COMPANY = "CBA.AX"
    START_DATE = "2020-01-01"
    END_DATE = "2024-07-01"
    N_STEPS = 50
    TEST_SIZE = 0.2
    DROPOUT = 0.2

    FULL_FEATURES = ['adjclose', 'volume', 'open', 'high', 'low']

    # Standard architecture for controlled experiments
    STANDARD_LAYERS = [
        {"type": "LSTM", "units": 50},
        {"type": "LSTM", "units": 50},
    ]

    # ================================================================
    # GROUP 1: MULTIVARIATE VS UNIVARIATE (single-step prediction)
    # ================================================================
    print("\n" + "=" * 60)
    print("GROUP 1: Feature Set Comparison (Single-Step)")
    print("=" * 60)

    feature_sets = {
        "Univariate (adjclose)": ["adjclose"],
        "Bivariate (adjclose+volume)": ["adjclose", "volume"],
        "Full multivariate": ["adjclose", "volume", "open", "high", "low"],
    }

    group1_results = []
    for name, features in feature_sets.items():
        print(f"\n--- {name} ({len(features)} features) ---")

        # Load data with this feature set
        data = load_data(
            ticker=COMPANY, start_date=START_DATE, end_date=END_DATE,
            n_steps=N_STEPS, scale=True, shuffle=True,
            lookup_step=1, n_future_steps=1,
            split_by_date=True, test_size=TEST_SIZE,
            feature_columns=features,
            save_local=True, load_local=True, local_path='data'
        )

        n_features = data['X_train'].shape[2]
        print(f"  Input shape: {data['X_train'].shape}")

        # Reset seeds for fair comparison
        np.random.seed(314)
        tf.random.set_seed(314)
        random.seed(314)

        model = create_model(
            input_shape=(N_STEPS, n_features),
            layer_configs=STANDARD_LAYERS.copy(),
            dropout_rate=DROPOUT,
            output_steps=1
        )

        start_time = time.time()
        model.fit(data["X_train"], data["y_train"],
                  epochs=25, batch_size=32,
                  validation_split=0.1, verbose=1)
        train_time = time.time() - start_time

        # Predict and inverse-transform
        y_pred = model.predict(data["X_test"])
        y_test = data["y_test"]

        y_pred_orig = data["column_scaler"]["adjclose"].inverse_transform(y_pred)
        y_test_orig = data["column_scaler"]["adjclose"].inverse_transform(
            y_test.reshape(-1, 1))

        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig.flatten(), y_pred_orig.flatten()))
        mape = np.mean(np.abs((y_test_orig.flatten() - y_pred_orig.flatten()) /
                               y_test_orig.flatten())) * 100
        r2 = r2_score(y_test_orig, y_pred_orig)

        print(f"  MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        print(f"  Training time: {train_time:.1f}s")

        group1_results.append({
            "Features": name,
            "N_Features": len(features),
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R²": round(r2, 4),
            "Time (s)": round(train_time, 1)
        })

    # Print Group 1 summary
    g1_df = pd.DataFrame(group1_results)
    print(f"\n{'='*60}")
    print("GROUP 1 SUMMARY: Feature Set Comparison")
    print(f"{'='*60}")
    print(g1_df.to_string(index=False))

    # ================================================================
    # GROUP 2: MULTISTEP PREDICTION (full features, vary k)
    # ================================================================
    print("\n" + "=" * 60)
    print("GROUP 2: Multistep Prediction (Full Features)")
    print("=" * 60)

    step_counts = [1, 3, 5, 7]
    group2_results = []

    for k in step_counts:
        print(f"\n--- {k}-step prediction ---")

        data = load_data(
            ticker=COMPANY, start_date=START_DATE, end_date=END_DATE,
            n_steps=N_STEPS, scale=True, shuffle=True,
            lookup_step=1, n_future_steps=k,
            split_by_date=True, test_size=TEST_SIZE,
            feature_columns=FULL_FEATURES,
            save_local=True, load_local=True, local_path='data'
        )

        n_features = data['X_train'].shape[2]
        print(f"  Input shape: {data['X_train'].shape}")
        print(f"  Target shape: {data['y_train'].shape}")

        np.random.seed(314)
        tf.random.set_seed(314)
        random.seed(314)

        model = create_model(
            input_shape=(N_STEPS, n_features),
            layer_configs=STANDARD_LAYERS.copy(),
            dropout_rate=DROPOUT,
            output_steps=k
        )

        start_time = time.time()
        model.fit(data["X_train"], data["y_train"],
                  epochs=25, batch_size=32,
                  validation_split=0.1, verbose=1)
        train_time = time.time() - start_time

        y_pred = model.predict(data["X_test"])
        y_test = data["y_test"]

        if k == 1:
            # Single step — same as Group 1
            y_pred_orig = data["column_scaler"]["adjclose"].inverse_transform(y_pred)
            y_test_orig = data["column_scaler"]["adjclose"].inverse_transform(
                y_test.reshape(-1, 1))
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_test_orig.flatten(), y_pred_orig.flatten()))
            mape = np.mean(np.abs((y_test_orig.flatten() - y_pred_orig.flatten()) /
                                   y_test_orig.flatten())) * 100
            r2 = r2_score(y_test_orig, y_pred_orig)

            group2_results.append({
                "Steps": k,
                "MAE ($)": round(mae, 2),
                "RMSE ($)": round(rmse, 2),
                "MAPE (%)": round(mape, 2),
                "R²": round(r2, 4),
                "Time (s)": round(train_time, 1)
            })
            print(f"  MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        else:
            # Multistep — use evaluate_multistep()
            eval_results = evaluate_multistep(
                y_test, y_pred, data["column_scaler"]["adjclose"])

            overall = eval_results['overall']
            group2_results.append({
                "Steps": k,
                "MAE ($)": overall['MAE ($)'],
                "RMSE ($)": overall['RMSE ($)'],
                "MAPE (%)": overall['MAPE (%)'],
                "R²": overall['R²'],
                "Time (s)": round(train_time, 1)
            })

            print(f"  Overall: MAE=${overall['MAE ($)']}, RMSE=${overall['RMSE ($)']}, "
                  f"MAPE={overall['MAPE (%)']:.2f}%, R²={overall['R²']:.4f}")

            # Print per-step breakdown
            print(f"  Per-step breakdown:")
            for step_result in eval_results['per_step']:
                print(f"    Step {step_result['step']}: "
                      f"MAE=${step_result['MAE ($)']}, "
                      f"MAPE={step_result['MAPE (%)']:.2f}%, "
                      f"R²={step_result['R²']:.4f}")

            # Plot trajectory and per-step error for k=5
            if k == 5:
                plot_multistep_trajectory(
                    y_test, y_pred, data["column_scaler"]["adjclose"],
                    n_trajectories=5,
                    title=f"Multistep Trajectories (k={k})",
                    save_path=f"plot_g2_trajectory_k{k}.png"
                )
                plot_per_step_error(
                    eval_results['per_step'],
                    metric='MAPE (%)',
                    title=f"Per-Step MAPE (k={k})",
                    save_path=f"plot_g2_per_step_error_k{k}.png"
                )

    # Print Group 2 summary
    g2_df = pd.DataFrame(group2_results)
    print(f"\n{'='*60}")
    print("GROUP 2 SUMMARY: Multistep Prediction")
    print(f"{'='*60}")
    print(g2_df.to_string(index=False))

    # ================================================================
    # GROUP 3: COMBINED MULTIVARIATE + MULTISTEP
    # ================================================================
    print("\n" + "=" * 60)
    print("GROUP 3: Combined Multivariate + Multistep (k=5)")
    print("=" * 60)

    group3_configs = [
        {
            "name": "LSTM 2x50",
            "layers": [{"type": "LSTM", "units": 50}, {"type": "LSTM", "units": 50}]
        },
        {
            "name": "LSTM 2x100",
            "layers": [{"type": "LSTM", "units": 100}, {"type": "LSTM", "units": 100}]
        },
        {
            "name": "GRU 2x50",
            "layers": [{"type": "GRU", "units": 50}, {"type": "GRU", "units": 50}]
        },
    ]

    K_COMBINED = 5
    group3_results = []

    # Load data once for Group 3 (all use same features and k)
    data = load_data(
        ticker=COMPANY, start_date=START_DATE, end_date=END_DATE,
        n_steps=N_STEPS, scale=True, shuffle=True,
        lookup_step=1, n_future_steps=K_COMBINED,
        split_by_date=True, test_size=TEST_SIZE,
        feature_columns=FULL_FEATURES,
        save_local=True, load_local=True, local_path='data'
    )
    n_features = data['X_train'].shape[2]

    for config in group3_configs:
        print(f"\n--- {config['name']} ---")

        np.random.seed(314)
        tf.random.set_seed(314)
        random.seed(314)

        model = create_model(
            input_shape=(N_STEPS, n_features),
            layer_configs=[cfg.copy() for cfg in config["layers"]],
            dropout_rate=DROPOUT,
            output_steps=K_COMBINED
        )

        start_time = time.time()
        model.fit(data["X_train"], data["y_train"],
                  epochs=25, batch_size=32,
                  validation_split=0.1, verbose=1)
        train_time = time.time() - start_time

        y_pred = model.predict(data["X_test"])
        y_test = data["y_test"]

        eval_results = evaluate_multistep(
            y_test, y_pred, data["column_scaler"]["adjclose"])

        overall = eval_results['overall']
        print(f"  Overall: MAE=${overall['MAE ($)']}, RMSE=${overall['RMSE ($)']}, "
              f"MAPE={overall['MAPE (%)']:.2f}%, R²={overall['R²']:.4f}")
        print(f"  Training time: {train_time:.1f}s")

        group3_results.append({
            "Architecture": config["name"],
            "MAE ($)": overall['MAE ($)'],
            "RMSE ($)": overall['RMSE ($)'],
            "MAPE (%)": overall['MAPE (%)'],
            "R²": overall['R²'],
            "Time (s)": round(train_time, 1)
        })

    # Print Group 3 summary
    g3_df = pd.DataFrame(group3_results)
    print(f"\n{'='*60}")
    print("GROUP 3 SUMMARY: Combined Multivariate + Multistep")
    print(f"{'='*60}")
    print(g3_df.to_string(index=False))

    # ================================================================
    # SAVE ALL RESULTS TO CSV
    # ================================================================
    all_results = []
    for r in group1_results:
        all_results.append({**r, "Group": "1-Features"})
    for r in group2_results:
        all_results.append({**r, "Group": "2-Multistep"})
    for r in group3_results:
        all_results.append({**r, "Group": "3-Combined"})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("experiment_results_v05.csv", index=False)
    print(f"\nAll results saved to 'experiment_results_v05.csv'")
    print("=" * 60)
