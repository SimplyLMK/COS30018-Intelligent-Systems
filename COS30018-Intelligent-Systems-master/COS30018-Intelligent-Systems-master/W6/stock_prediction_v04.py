# File: stock_prediction_v04.py
# Authors: Le Minh Kha (based on work by Bao Vo, Cheong Koo, and x4nth055)
# Version: 0.4 - Machine Learning 1 (Dynamic Model Builder + Experiments)
#
# This version builds on v0.3 by adding:
# - create_model() function: dynamically builds models from config lists
#   supporting LSTM, GRU, SimpleRNN, and Dense layers
# - Experiment runner: systematically tests different model configurations
#   (network type, depth, width, training hyperparameters)
# All data loading and visualization functions remain from v0.3.

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
# NEW IN v0.4: DYNAMIC MODEL BUILDER (Task C.4)
#------------------------------------------------------------------------------
def create_model(input_shape, layer_configs, dropout_rate=0.2,
                 loss="mean_absolute_error", optimizer="adam"):
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
                {"type": "Dense", "units": 1}
            ],
            dropout_rate=0.2
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

    # --- Auto-append Dense(1) if last layer is not Dense ---
    # Ensures the model always ends with a single output for regression
    if layer_configs[-1]["type"] != "Dense":
        layer_configs = layer_configs + [{"type": "Dense", "units": 1}]

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
# MAIN FUNCTION: LOAD AND PROCESS DATA
#------------------------------------------------------------------------------
def load_data(ticker,
              start_date=None,
              end_date=None,
              n_steps=50,
              scale=True,
              shuffle=True,
              lookup_step=1,
              split_by_date=True,
              test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              save_local=False,
              load_local=False,
              local_path='data'):
    """
    Load stock data from Yahoo Finance and process it for LSTM model training.
    (Same as v0.2 - see v0.2 for full documentation)
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

    # STEP 6: Create target column
    df['future'] = df['adjclose'].shift(-lookup_step)

    # STEP 7: Save last sequence
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # STEP 8: Drop NaN
    df.dropna(inplace=True)

    # STEP 9: Create sequences
    sequence_data = []
    sequences = deque(maxlen=n_steps)

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


#==============================================================================
# FROM v0.3: VISUALIZATION FUNCTIONS (Task C.3)
#==============================================================================

#------------------------------------------------------------------------------
# TASK C.3.1: CANDLESTICK CHART
#------------------------------------------------------------------------------
def plot_candlestick(df, n_days=1, ticker="Stock", start_date=None, end_date=None):
    """
    Display stock market financial data using a candlestick chart.

    A candlestick chart shows four key prices for each time period:
    - Open: the price at market open
    - High: the highest price during the period
    - Low: the lowest price during the period
    - Close: the price at market close

    Each "candle" has:
    - A body (rectangle): shows Open-to-Close range
      - Green/hollow body: Close > Open (price went UP)
      - Red/filled body: Close < Open (price went DOWN)
    - Wicks/shadows (thin lines): show High and Low extremes

    This function uses the mplfinance library, which is specifically designed
    for financial data visualization and works directly with OHLC data.

    Reference: https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing stock data. Must have columns:
        'open', 'high', 'low', 'close' (or 'adjclose').
        The index must be a DatetimeIndex (dates).

    n_days : int, default=1
        Number of trading days each candlestick represents.
        - n_days=1: each candle = 1 trading day (daily chart)
        - n_days=5: each candle = 5 trading days (~weekly chart)
        - n_days=20: each candle = 20 trading days (~monthly chart)
        When n_days > 1, the data is resampled by aggregating:
          Open  -> first value in the period
          High  -> maximum value in the period
          Low   -> minimum value in the period
          Close -> last value in the period
          Volume -> sum of all values in the period

    ticker : str, default="Stock"
        Stock ticker symbol, used in the chart title.

    start_date : str, optional
        Start date for filtering the data ('YYYY-MM-DD' format).
        If None, uses the earliest date in the DataFrame.

    end_date : str, optional
        End date for filtering the data ('YYYY-MM-DD' format).
        If None, uses the latest date in the DataFrame.

    Returns:
    --------
    None (displays the chart)
    """

    # ------------------------------------------------------------------
    # Step 1: Create a working copy to avoid modifying the original data
    # ------------------------------------------------------------------
    data = df.copy()

    # ------------------------------------------------------------------
    # Step 2: Ensure the index is a DatetimeIndex
    # ------------------------------------------------------------------
    # mplfinance requires a DatetimeIndex to properly format the x-axis
    # with date labels and handle time-based operations
    if not isinstance(data.index, pd.DatetimeIndex):
        # If the index is not already datetime, try to convert it
        data.index = pd.to_datetime(data.index)

    # Explicitly set the index name to 'Date' (mplfinance convention)
    data.index.name = 'Date'

    # ------------------------------------------------------------------
    # Step 3: Standardize column names for mplfinance
    # ------------------------------------------------------------------
    # mplfinance expects specific column names: 'Open', 'High', 'Low',
    # 'Close', and optionally 'Volume' (title case)
    # Our data uses lowercase names, so we need to create a mapping

    # Build a new DataFrame with only the columns mplfinance needs.
    # We must avoid renaming both 'close' and 'adjclose' to 'Close',
    # which would create duplicate columns and cause a type error.
    # Strategy: prefer 'adjclose' over 'close' for the Close column,
    # because adjusted close accounts for stock splits and dividends.

    cols_lower = {col: col.lower() for col in data.columns}
    ohlcv = {}

    # Map each required mplfinance column to the best available source
    for col, col_lower in cols_lower.items():
        if col_lower == 'open':
            ohlcv['Open'] = data[col]
        elif col_lower == 'high':
            ohlcv['High'] = data[col]
        elif col_lower == 'low':
            ohlcv['Low'] = data[col]
        elif col_lower == 'volume':
            ohlcv['Volume'] = data[col]

    # For Close: prefer adjclose (adjusted for splits/dividends) over close
    if 'adjclose' in cols_lower.values():
        # Find the original column name that maps to 'adjclose'
        adjclose_col = [c for c, cl in cols_lower.items() if cl == 'adjclose'][0]
        ohlcv['Close'] = data[adjclose_col]
    elif 'close' in cols_lower.values():
        close_col = [c for c, cl in cols_lower.items() if cl == 'close'][0]
        ohlcv['Close'] = data[close_col]

    # Build a clean DataFrame with only OHLCV columns and the same index
    data = pd.DataFrame(ohlcv, index=data.index)

    # ------------------------------------------------------------------
    # Step 4: Filter by date range if specified
    # ------------------------------------------------------------------
    # loc[] with slice allows selecting rows within a date range
    # This is inclusive on both ends for DatetimeIndex
    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]

    # ------------------------------------------------------------------
    # Step 5: Resample data if n_days > 1
    # ------------------------------------------------------------------
    # When n_days > 1, we aggregate multiple trading days into one candle
    # This gives a higher-level view of price trends
    #
    # resample() groups data by time periods, similar to groupby for dates
    # f'{n_days}D' creates a frequency string: '5D' = 5 calendar days
    # However, using business days ('B') would be more accurate for trading
    #
    # Aggregation rules (standard OHLCV resampling):
    # - Open:   'first' -> opening price of the first day in the period
    # - High:   'max'   -> highest price across all days in the period
    # - Low:    'min'   -> lowest price across all days in the period
    # - Close:  'last'  -> closing price of the last day in the period
    # - Volume: 'sum'   -> total volume traded across all days

    if n_days > 1:
        # Define how each column should be aggregated
        # agg() applies different functions to different columns
        ohlc_dict = {
            'Open': 'first',    # First opening price in the period
            'High': 'max',      # Maximum high across the period
            'Low': 'min',       # Minimum low across the period
            'Close': 'last',    # Last closing price in the period
        }

        # Only include Volume in aggregation if it exists in the data
        if 'Volume' in data.columns:
            ohlc_dict['Volume'] = 'sum'  # Total volume for the period

        # resample(f'{n_days}D') groups data into n_days calendar-day bins
        # .agg(ohlc_dict) applies the specified aggregation to each column
        # .dropna() removes any periods with no trading data (weekends/holidays)
        data = data.resample(f'{n_days}D').agg(ohlc_dict).dropna()

    # ------------------------------------------------------------------
    # Step 6: Ensure required columns exist
    # ------------------------------------------------------------------
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {list(data.columns)}. "
                f"Ensure the DataFrame has OHLC data.")

    # ------------------------------------------------------------------
    # Step 7: Configure the chart style
    # ------------------------------------------------------------------
    # mpf.make_marketcolors() defines the color scheme for candlesticks
    #   up: color for bullish candles (close > open, price went up)
    #   down: color for bearish candles (close < open, price went down)
    #   wick: color of the thin lines showing high/low extremes
    #   edge: color of the candle body border
    #   volume: color scheme for volume bars ('in' = match candle colors)
    market_colors = mpf.make_marketcolors(
        up='green',       # Bullish candle color (price increased)
        down='red',       # Bearish candle color (price decreased)
        wick='inherit',   # Wick color inherits from up/down colors
        edge='inherit',   # Edge color inherits from up/down colors
        volume='in'       # Volume bars colored to match candle direction
    )

    # mpf.make_mpf_style() creates a complete chart style
    #   marketcolors: the color scheme defined above
    #   gridstyle: style of background grid lines ('--' = dashed)
    #   gridaxis: which axes get grid lines ('both' = x and y)
    chart_style = mpf.make_mpf_style(
        marketcolors=market_colors,
        gridstyle='--',   # Dashed grid lines for readability
        gridaxis='both'   # Show grid on both x-axis and y-axis
    )

    # ------------------------------------------------------------------
    # Step 8: Create and display the candlestick chart
    # ------------------------------------------------------------------
    # mpf.plot() is the main plotting function from mplfinance
    #
    # Parameters:
    #   data: DataFrame with OHLCV data and DatetimeIndex
    #   type: chart type - 'candle' for candlestick chart
    #         (other options: 'ohlc', 'line', 'renko', 'pnf')
    #   style: visual style object created by make_mpf_style()
    #   title: chart title displayed at the top
    #   ylabel: label for the y-axis (price axis)
    #   volume: if True, show volume bars in a subplot below the price chart
    #   figsize: (width, height) in inches for the figure
    #   tight_layout: if True, auto-adjust subplot spacing to prevent overlap

    # Build the title string with the n-day grouping information
    if n_days == 1:
        title_str = f"{ticker} Candlestick Chart (Daily)"
    else:
        title_str = f"{ticker} Candlestick Chart ({n_days}-Day Candles)"

    # Check if Volume column exists to decide whether to show volume subplot
    show_volume = 'Volume' in data.columns

    mpf.plot(
        data,                       # OHLCV DataFrame with DatetimeIndex
        type='candle',              # Chart type: candlestick
        style=chart_style,          # Visual style (colors, grid)
        title=title_str,            # Chart title
        ylabel='Price ($)',         # Y-axis label
        volume=show_volume,         # Show volume subplot if data available
        ylabel_lower='Volume',      # Label for volume subplot y-axis
        figsize=(14, 8),            # Figure size: 14 inches wide, 8 tall
        tight_layout=True           # Auto-adjust spacing between subplots
    )


#------------------------------------------------------------------------------
# TASK C.3.2: BOXPLOT CHART
#------------------------------------------------------------------------------
def plot_boxplot(df, n_days=5, column='adjclose', ticker="Stock",
                 start_date=None, end_date=None):
    """
    Display stock market financial data using a boxplot chart with a
    moving window of n consecutive trading days.

    A boxplot (box-and-whisker plot) displays the distribution of data
    through five key statistics:
    - Minimum (lower whisker): smallest non-outlier value
    - Q1 (25th percentile / lower edge of box): 25% of values are below
    - Median (line inside box): the middle value (50th percentile)
    - Q3 (75th percentile / upper edge of box): 75% of values are below
    - Maximum (upper whisker): largest non-outlier value
    - Outliers: points beyond 1.5 * IQR from Q1/Q3 (shown as dots)

    Where IQR = Q3 - Q1 (Interquartile Range).

    This function divides the stock data into non-overlapping windows of
    n consecutive trading days and creates one boxplot per window.
    This is useful for seeing how price distribution changes over time.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing stock data with a DatetimeIndex.

    n_days : int, default=5
        Number of consecutive trading days per boxplot window.
        - n_days=5:  ~1 trading week per box
        - n_days=20: ~1 trading month per box
        Each box summarizes the distribution of 'column' values
        within that window of n trading days.

    column : str, default='adjclose'
        Which column to plot. Common choices:
        - 'adjclose': adjusted closing price
        - 'close': raw closing price
        - 'volume': trading volume

    ticker : str, default="Stock"
        Stock ticker symbol, used in the chart title.

    start_date : str, optional
        Start date for filtering ('YYYY-MM-DD' format).

    end_date : str, optional
        End date for filtering ('YYYY-MM-DD' format).

    Returns:
    --------
    None (displays the chart)
    """

    # ------------------------------------------------------------------
    # Step 1: Create a working copy to avoid modifying the original
    # ------------------------------------------------------------------
    data = df.copy()

    # ------------------------------------------------------------------
    # Step 2: Ensure the index is a DatetimeIndex
    # ------------------------------------------------------------------
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # ------------------------------------------------------------------
    # Step 3: Filter by date range if specified
    # ------------------------------------------------------------------
    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]

    # ------------------------------------------------------------------
    # Step 4: Validate that the requested column exists
    # ------------------------------------------------------------------
    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found. Available: {list(data.columns)}")

    # ------------------------------------------------------------------
    # Step 5: Divide data into non-overlapping windows of n trading days
    # ------------------------------------------------------------------
    # We collect the values for each window along with a label (the date
    # range) for the x-axis tick marks.
    #
    # range(0, len(data), n_days) generates start indices:
    #   0, n_days, 2*n_days, 3*n_days, ...
    # iloc[i:i+n_days] selects rows from index i to i+n_days (exclusive)

    window_data = []   # List of arrays, one per window
    window_labels = [] # List of date-range strings for x-axis labels

    for i in range(0, len(data), n_days):
        # Extract the slice of data for this window
        window = data.iloc[i:i + n_days]

        # Skip incomplete windows at the end (fewer than n_days rows)
        # This avoids misleading boxplots based on insufficient data
        if len(window) < n_days:
            break

        # Get the column values for this window as a numpy array
        # .values converts the pandas Series to a numpy array
        window_data.append(window[column].values)

        # Create a label showing the date range for this window
        # strftime('%Y-%m-%d') formats a datetime as 'YYYY-MM-DD'
        label_start = window.index[0].strftime('%Y-%m-%d')
        label_end = window.index[-1].strftime('%Y-%m-%d')
        window_labels.append(f"{label_start}\nto\n{label_end}")

    # ------------------------------------------------------------------
    # Step 6: Check that we have data to plot
    # ------------------------------------------------------------------
    if len(window_data) == 0:
        print("Warning: Not enough data to create boxplots with the given "
              f"window size of {n_days} days.")
        return

    # ------------------------------------------------------------------
    # Step 7: Create the boxplot figure
    # ------------------------------------------------------------------
    # Determine figure width based on number of boxes
    # min/max bounds prevent the figure from being too narrow or too wide
    fig_width = max(10, min(len(window_data) * 0.8, 30))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # ------------------------------------------------------------------
    # Step 8: Draw the boxplots
    # ------------------------------------------------------------------
    # ax.boxplot() creates box-and-whisker plots
    #
    # Parameters:
    #   window_data: list of arrays, one boxplot per array
    #   labels: x-axis tick labels (date ranges)
    #   patch_artist: if True, fills box bodies with color (not just outlines)
    #   boxprops: styling for the box body (face color, edge color)
    #   medianprops: styling for the median line inside each box
    #   whiskerprops: styling for the whisker lines (extending to min/max)
    #   capprops: styling for the caps at the end of whiskers (horizontal bars)
    #   flierprops: styling for outlier points (beyond 1.5*IQR)
    #   widths: width of each box (in x-axis units)

    bp = ax.boxplot(
        window_data,
        labels=window_labels,
        patch_artist=True,              # Fill boxes with color
        boxprops=dict(
            facecolor='lightblue',      # Box fill color
            edgecolor='black',          # Box border color
            linewidth=1.2               # Box border thickness
        ),
        medianprops=dict(
            color='red',                # Median line color
            linewidth=2                 # Median line thickness
        ),
        whiskerprops=dict(
            color='black',              # Whisker line color
            linewidth=1.0               # Whisker line thickness
        ),
        capprops=dict(
            color='black',              # Cap (end of whisker) color
            linewidth=1.0               # Cap line thickness
        ),
        flierprops=dict(
            marker='o',                 # Outlier marker shape (circle)
            markerfacecolor='gray',     # Outlier fill color
            markersize=4,               # Outlier marker size
            alpha=0.6                   # Outlier transparency (0=invisible, 1=opaque)
        ),
        widths=0.6                      # Box width relative to spacing
    )

    # ------------------------------------------------------------------
    # Step 9: Format the chart
    # ------------------------------------------------------------------
    # Set title with information about the window size
    ax.set_title(
        f"{ticker} {column.title()} - Boxplot "
        f"({n_days}-Day Moving Window)",
        fontsize=14,
        fontweight='bold'
    )

    ax.set_ylabel(f'{column.title()} ($)', fontsize=12)
    ax.set_xlabel('Trading Period', fontsize=12)

    # Rotate x-axis labels to prevent overlapping
    # ha='right' aligns the right edge of rotated text to the tick position
    plt.xticks(rotation=45, ha='right', fontsize=7)

    # Enable grid on the y-axis only for easier value reading
    # alpha controls the transparency of the grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # tight_layout() auto-adjusts margins so labels don't get cut off
    plt.tight_layout()
    plt.show()


#==============================================================================
# MAIN EXECUTION
#==============================================================================
if __name__ == "__main__":

    #--------------------------------------------------------------------------
    # CONFIGURATION PARAMETERS
    #--------------------------------------------------------------------------
    COMPANY = "CBA.AX"
    START_DATE = "2020-01-01"
    END_DATE = "2024-07-01"

    N_STEPS = 50
    LOOKUP_STEP = 1
    TEST_SIZE = 0.2
    FEATURE_COLUMNS = ['adjclose', 'volume', 'open', 'high', 'low']

    DROPOUT = 0.2  # Dropout rate for experiments

    #--------------------------------------------------------------------------
    # LOAD AND PROCESS DATA
    #--------------------------------------------------------------------------
    print("=" * 60)
    print("STOCK PREDICTION v0.4 - Machine Learning 1")
    print("Dynamic Model Builder + Experiments")
    print("=" * 60)
    print(f"\nLoading data for {COMPANY}...")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Sequence length: {N_STEPS} days")
    print(f"Prediction horizon: {LOOKUP_STEP} day(s) ahead")
    print(f"Test size: {TEST_SIZE*100}%")
    print(f"Split method: By Date")

    data = load_data(
        ticker=COMPANY,
        start_date=START_DATE,
        end_date=END_DATE,
        n_steps=N_STEPS,
        scale=True,
        shuffle=True,
        lookup_step=LOOKUP_STEP,
        split_by_date=True,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS,
        save_local=True,
        load_local=True,
        local_path='data'
    )

    print(f"\n{'='*60}")
    print("DATA LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total samples in original data: {len(data['df'])}")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Testing samples: {len(data['X_test'])}")
    print(f"Feature shape: {data['X_train'].shape}")

    # Extract dimensions for create_model()
    n_steps = data['X_train'].shape[1]
    n_features = data['X_train'].shape[2]

    #==========================================================================
    # NEW IN v0.4: EXPERIMENT GRID (Task C.4)
    #==========================================================================
    # Define all experiments as a list of dicts. Each dict specifies:
    #   - name: human-readable identifier
    #   - layers: list of layer configs (Dense(1) is appended by create_model)
    #   - epochs: number of training epochs
    #   - batch_size: mini-batch size for training

    experiments = []

    # --- Experiment 1: Network Type Comparison ---
    # Isolates the effect of the recurrent cell type (LSTM vs GRU vs SimpleRNN)
    # All other hyperparameters are held constant.
    for cell_type in ["LSTM", "GRU", "SimpleRNN"]:
        experiments.append({
            "name": f"Exp1-{cell_type}-2x50",
            "group": "1-CellType",
            "layers": [
                {"type": cell_type, "units": 50},
                {"type": cell_type, "units": 50},
                {"type": "Dense", "units": 1}
            ],
            "epochs": 25,
            "batch_size": 32
        })

    # --- Experiment 2: Depth Comparison ---
    # Isolates the effect of stacking more layers (1 vs 2 vs 3 LSTM layers)
    for depth, label in [(1, "Shallow"), (2, "Medium"), (3, "Deep")]:
        layers = [{"type": "LSTM", "units": 50} for _ in range(depth)]
        layers.append({"type": "Dense", "units": 1})
        experiments.append({
            "name": f"Exp2-{label}-{depth}L",
            "group": "2-Depth",
            "layers": layers,
            "epochs": 25,
            "batch_size": 32
        })

    # --- Experiment 3: Width Comparison ---
    # Isolates the effect of layer width (25 vs 50 vs 100 units per layer)
    for units, label in [(25, "Narrow"), (50, "Medium"), (100, "Wide")]:
        experiments.append({
            "name": f"Exp3-{label}-{units}u",
            "group": "3-Width",
            "layers": [
                {"type": "LSTM", "units": units},
                {"type": "LSTM", "units": units},
                {"type": "Dense", "units": 1}
            ],
            "epochs": 25,
            "batch_size": 32
        })

    # --- Experiment 4: Training Hyperparameters ---
    # Varies epochs and batch size while keeping architecture constant
    for config_label, epochs, batch in [("A", 25, 32), ("B", 50, 32),
                                         ("C", 25, 16), ("D", 50, 64)]:
        experiments.append({
            "name": f"Exp4-{config_label}-e{epochs}b{batch}",
            "group": "4-HyperParams",
            "layers": [
                {"type": "LSTM", "units": 50},
                {"type": "LSTM", "units": 50},
                {"type": "Dense", "units": 1}
            ],
            "epochs": epochs,
            "batch_size": batch
        })

    #--------------------------------------------------------------------------
    # RUN ALL EXPERIMENTS
    #--------------------------------------------------------------------------
    results = []

    for idx, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Running experiment {idx+1}/{len(experiments)}: {exp['name']}")
        print(f"Group: {exp['group']}")
        print(f"{'='*60}")

        # Reset seeds before each experiment for fair comparison
        np.random.seed(314)
        tf.random.set_seed(314)
        random.seed(314)

        # Build model dynamically using create_model()
        model = create_model(
            input_shape=(n_steps, n_features),
            layer_configs=exp["layers"],
            dropout_rate=DROPOUT,
            loss="mean_absolute_error",
            optimizer="adam"
        )

        # Print model summary for the first experiment (for screenshots)
        if idx == 0:
            print("\nModel Summary (first experiment):")
            model.summary()

        # Train the model and measure training time
        start_time = time.time()
        model.fit(
            data["X_train"], data["y_train"],
            epochs=exp["epochs"],
            batch_size=exp["batch_size"],
            validation_split=0.1,
            verbose=1
        )
        train_time = time.time() - start_time

        # Predict on test set
        y_pred = model.predict(data["X_test"])
        y_test = data["y_test"]

        # Inverse-transform predictions and actuals back to original scale
        y_pred_original = data["column_scaler"]["adjclose"].inverse_transform(y_pred)
        y_test_original = data["column_scaler"]["adjclose"].inverse_transform(
            y_test.reshape(-1, 1))

        y_test_flat = y_test_original.flatten()
        y_pred_flat = y_pred_original.flatten()

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100
        r2 = r2_score(y_test_flat, y_pred_flat)

        # Print individual experiment results
        print(f"\n--- Results for {exp['name']} ---")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        print(f"  Training time: {train_time:.1f}s")

        # Store results for summary table
        results.append({
            "Experiment": exp["name"],
            "Group": exp["group"],
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R²": round(r2, 4),
            "Time (s)": round(train_time, 1)
        })

    #--------------------------------------------------------------------------
    # PRINT SUMMARY TABLE
    #--------------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))

    # Print per-group summaries for easier comparison
    for group_name in results_df["Group"].unique():
        group_df = results_df[results_df["Group"] == group_name]
        print(f"\n--- {group_name} ---")
        print(group_df.to_string(index=False))

    # Save results to CSV for later analysis / report
    results_df.to_csv("experiment_results_v04.csv", index=False)
    print(f"\nResults saved to 'experiment_results_v04.csv'")
    print("=" * 80)
