# File: stock_prediction_v03.py
# Authors: Le Minh Kha (based on work by Bao Vo, Cheong Koo, and x4nth055)
# Version: 0.3 - Data Processing 2 (Visualization)
#
# This version builds on v0.2 by adding:
# - Candlestick chart visualization with n-day grouping
# - Boxplot chart visualization for moving window of n trading days
# All data loading and model functions remain from v0.2.

#------------------------------------------------------------------------------
# IMPORT LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import yfinance as yf
import tensorflow as tf
import random
import mplfinance as mpf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from collections import deque

#------------------------------------------------------------------------------
# SET RANDOM SEEDS FOR REPRODUCIBILITY
#------------------------------------------------------------------------------
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


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


#------------------------------------------------------------------------------
# FUNCTION: CREATE LSTM MODEL
#------------------------------------------------------------------------------
def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2,
                 dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop",
                 bidirectional=False):
    """
    Create an LSTM model for stock price prediction.
    (Same as v0.2 - see v0.2 for full documentation)
    """
    model = Sequential()

    for i in range(n_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(
                    cell(units, return_sequences=True),
                    input_shape=(sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True,
                             input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))

        model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


#==============================================================================
# NEW IN v0.3: VISUALIZATION FUNCTIONS (Task C.3)
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

    N_LAYERS = 2
    UNITS = 256
    DROPOUT = 0.3
    BIDIRECTIONAL = False

    EPOCHS = 25
    BATCH_SIZE = 32

    #--------------------------------------------------------------------------
    # LOAD AND PROCESS DATA
    #--------------------------------------------------------------------------
    print("=" * 60)
    print("STOCK PREDICTION v0.3 - Data Processing 2 (Visualization)")
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

    #--------------------------------------------------------------------------
    # NEW IN v0.3: VISUALIZATION - CANDLESTICK CHARTS
    #--------------------------------------------------------------------------
    # Use the unscaled original DataFrame (data['df']) for visualization
    # The original DataFrame has actual prices, not normalized values

    print(f"\n{'='*60}")
    print("VISUALIZATION: CANDLESTICK CHARTS (Task C.3.1)")
    print(f"{'='*60}")

    # Candlestick chart - Daily (n_days=1)
    print("\n[1/3] Displaying daily candlestick chart...")
    plot_candlestick(
        df=data['df'],
        n_days=1,
        ticker=COMPANY,
        start_date="2024-01-01",    # Show last 6 months for readability
        end_date="2024-07-01"
    )

    # Candlestick chart - Weekly (n_days=5)
    print("[2/3] Displaying weekly (5-day) candlestick chart...")
    plot_candlestick(
        df=data['df'],
        n_days=5,
        ticker=COMPANY,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Candlestick chart - Monthly (n_days=20)
    print("[3/3] Displaying monthly (20-day) candlestick chart...")
    plot_candlestick(
        df=data['df'],
        n_days=20,
        ticker=COMPANY,
        start_date=START_DATE,
        end_date=END_DATE
    )

    #--------------------------------------------------------------------------
    # NEW IN v0.3: VISUALIZATION - BOXPLOT CHARTS
    #--------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("VISUALIZATION: BOXPLOT CHARTS (Task C.3.2)")
    print(f"{'='*60}")

    # Boxplot - 5-day window (~weekly)
    print("\n[1/2] Displaying 5-day window boxplot (adjclose)...")
    plot_boxplot(
        df=data['df'],
        n_days=5,
        column='adjclose',
        ticker=COMPANY,
        start_date="2024-01-01",
        end_date="2024-07-01"
    )

    # Boxplot - 20-day window (~monthly)
    print("[2/2] Displaying 20-day window boxplot (adjclose)...")
    plot_boxplot(
        df=data['df'],
        n_days=20,
        column='adjclose',
        ticker=COMPANY,
        start_date=START_DATE,
        end_date=END_DATE
    )

    #--------------------------------------------------------------------------
    # BUILD THE MODEL
    #--------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}")

    model = create_model(
        sequence_length=N_STEPS,
        n_features=len(FEATURE_COLUMNS),
        units=UNITS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        loss="mean_absolute_error",
        optimizer="adam"
    )

    model.summary()

    #--------------------------------------------------------------------------
    # TRAIN THE MODEL
    #--------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")

    history = model.fit(
        data["X_train"],
        data["y_train"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data["X_test"], data["y_test"]),
        verbose=1
    )

    #--------------------------------------------------------------------------
    # EVALUATE AND PREDICT
    #--------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EVALUATING MODEL")
    print(f"{'='*60}")

    y_pred = model.predict(data["X_test"])
    y_test = data["y_test"]

    y_pred_original = data["column_scaler"]["adjclose"].inverse_transform(y_pred)
    y_test_original = data["column_scaler"]["adjclose"].inverse_transform(
        y_test.reshape(-1, 1))

    #--------------------------------------------------------------------------
    # CALCULATE PERFORMANCE METRICS
    #--------------------------------------------------------------------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_test_flat = y_test_original.flatten()
    y_pred_flat = y_pred_original.flatten()

    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, y_pred_flat)
    mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100

    print(f"\nPERFORMANCE METRICS - v0.3")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE):      ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Squared Error (MSE):       {mse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R² Score:                       {r2:.4f}")
    print("=" * 60)

    #--------------------------------------------------------------------------
    # PLOT RESULTS
    #--------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(y_test_flat, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(y_pred_flat, label='Predicted Price', color='red', alpha=0.7)
    plt.title(f'{COMPANY} Stock Price Prediction\n'
              f'MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}')
    plt.xlabel('Time (Trading Days)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('v03_results.png', dpi=150)
    print(f"\nResults saved to 'v03_results.png'")
    plt.show()

    #--------------------------------------------------------------------------
    # PREDICT NEXT DAY
    #--------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FUTURE PREDICTION")
    print(f"{'='*60}")

    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    prediction = model.predict(last_sequence)
    predicted_price = data["column_scaler"]["adjclose"].inverse_transform(
        prediction)[0][0]

    print(f"Predicted price for next trading day: ${predicted_price:.2f}")
    print("=" * 60)
