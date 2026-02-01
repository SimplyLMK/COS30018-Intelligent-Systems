# File: stock_prediction_v02.py
# Authors: [Your Name] (based on work by Bao Vo, Cheong Koo, and x4nth055)
# Date: [Current Date]
# Version: 0.2 - Enhanced Data Processing
#
# This version improves upon v0.1 by implementing:
# - Multiple feature support (Open, High, Low, Close, Adj Close, Volume)
# - Flexible date range specification
# - NaN handling
# - Multiple train/test split methods (by date or random)
# - Local data caching to avoid repeated downloads
# - Feature scaling with scaler storage for inverse transformation

#------------------------------------------------------------------------------
# IMPORT LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import yfinance as yf
import tensorflow as tf
import random

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from collections import deque

#------------------------------------------------------------------------------
# SET RANDOM SEEDS FOR REPRODUCIBILITY
#------------------------------------------------------------------------------
# Setting seeds ensures that the random number generators produce the same
# sequence of random numbers each time the code is run. This is crucial for:
# 1. Debugging - getting consistent results helps identify issues
# 2. Reproducibility - others can replicate your exact results
# 3. Comparison - fair comparison between different model configurations

np.random.seed(314)          # Seed for NumPy random operations
tf.random.set_seed(314)      # Seed for TensorFlow random operations
random.seed(314)             # Seed for Python's built-in random module


#------------------------------------------------------------------------------
# HELPER FUNCTION: SHUFFLE TWO ARRAYS IN UNISON
#------------------------------------------------------------------------------
def shuffle_in_unison(a, b):
    """
    Shuffle two arrays in the same way (maintaining correspondence).

    Why is this needed?
    - When we have features (X) and labels (y), they must stay aligned
    - If we shuffle X and y separately, the relationship between input
      features and their corresponding labels would be broken
    - This function captures the random state after shuffling 'a',
      then resets to that same state before shuffling 'b'
    - Result: both arrays are shuffled identically

    Parameters:
        a (np.array): First array (typically X - features)
        b (np.array): Second array (typically y - labels)
    """
    # Get the current state of the random number generator
    # This state is like a "snapshot" of where the RNG is in its sequence
    state = np.random.get_state()

    # Shuffle array 'a' - this changes the RNG state
    np.random.shuffle(a)

    # Reset the RNG to the captured state (before shuffling 'a')
    np.random.set_state(state)

    # Shuffle array 'b' - using the same RNG state produces identical shuffle
    np.random.shuffle(b)


#------------------------------------------------------------------------------
# MAIN FUNCTION: LOAD AND PROCESS DATA
#------------------------------------------------------------------------------
def load_data(ticker,
              start_date=None,           # NEW: Requirement (a) - specify start date
              end_date=None,             # NEW: Requirement (a) - specify end date
              n_steps=50,
              scale=True,
              shuffle=True,
              lookup_step=1,
              split_by_date=True,        # Requirement (c) - split method
              test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              save_local=False,          # NEW: Requirement (d) - save data locally
              load_local=False,          # NEW: Requirement (d) - load from local
              local_path='data'):        # NEW: Requirement (d) - local storage path
    """
    Load stock data from Yahoo Finance and process it for LSTM model training.

    This function handles the complete data pipeline:
    1. Data acquisition (from Yahoo Finance or local cache)
    2. Data cleaning (NaN handling)
    3. Feature scaling (normalization)
    4. Sequence creation (for time series)
    5. Train/test splitting

    Parameters:
    -----------
    ticker : str or pd.DataFrame
        Stock ticker symbol (e.g., 'AAPL', 'CBA.AX') or pre-loaded DataFrame

    start_date : str, optional (NEW - Requirement a)
        Start date for data in 'YYYY-MM-DD' format
        If None, downloads all available data

    end_date : str, optional (NEW - Requirement a)
        End date for data in 'YYYY-MM-DD' format
        If None, downloads up to current date

    n_steps : int, default=50
        Number of time steps (days) to look back for each prediction
        This is the "window size" - how many past days the model sees

    scale : bool, default=True
        Whether to scale features to range [0, 1] using MinMaxScaler
        Scaling is important because:
        - Neural networks work better with normalized inputs
        - Features with larger values won't dominate training

    shuffle : bool, default=True
        Whether to shuffle the training data
        Shuffling helps prevent the model from learning order-dependent patterns

    lookup_step : int, default=1
        How many days ahead to predict
        lookup_step=1 means predict tomorrow's price
        lookup_step=5 means predict price 5 days from now

    split_by_date : bool, default=True (Requirement c)
        If True: split chronologically (first 80% train, last 20% test)
        If False: split randomly (random 80/20 split)
        Date-based splitting is more realistic for time series

    test_size : float, default=0.2 (Requirement c)
        Proportion of data to use for testing (0.2 = 20%)

    feature_columns : list, default=['adjclose', 'volume', 'open', 'high', 'low']
        Which columns to use as features for prediction
        Using multiple features can capture more market dynamics

    save_local : bool, default=False (NEW - Requirement d)
        If True, save downloaded data to local CSV file
        Useful for offline work or faster subsequent runs

    load_local : bool, default=False (NEW - Requirement d)
        If True, attempt to load data from local CSV first
        Falls back to downloading if local file doesn't exist

    local_path : str, default='data' (NEW - Requirement d)
        Directory path for local data storage

    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'df': Original DataFrame with all data
        - 'column_scaler': Dictionary of MinMaxScaler objects for each feature
        - 'X_train', 'y_train': Training features and labels
        - 'X_test', 'y_test': Testing features and labels
        - 'last_sequence': Most recent sequence for future prediction
        - 'test_df': DataFrame of test data with dates
    """

    #--------------------------------------------------------------------------
    # STEP 1: DATA ACQUISITION
    #--------------------------------------------------------------------------
    # Handle different input types for the ticker parameter

    if isinstance(ticker, str):
        # ticker is a string (stock symbol) - need to download data

        #----------------------------------------------------------------------
        # NEW: LOCAL DATA LOADING (Requirement d)
        #----------------------------------------------------------------------
        # Construct the local file path for this ticker
        local_file = os.path.join(local_path, f"{ticker}_{start_date}_{end_date}.csv")

        if load_local and os.path.exists(local_file):
            # Load from local cache - faster than downloading
            print(f"Loading data from local file: {local_file}")
            df = pd.read_csv(local_file, index_col=0, parse_dates=True)
            df.index.name = 'date'
        else:
            # Download from Yahoo Finance using yfinance library
            print(f"Downloading data for {ticker} from Yahoo Finance...")

            # yf.download() parameters:
            # - ticker: stock symbol
            # - start: start date (None = earliest available)
            # - end: end date (None = today)
            # - progress: show download progress bar
            # - auto_adjust: whether to adjust prices for splits/dividends
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
                # No dates specified - download all available data
                df = yf.download(ticker, period="max",
                               progress=False, auto_adjust=False)

            #------------------------------------------------------------------
            # HANDLE MULTIINDEX COLUMNS
            #------------------------------------------------------------------
            # yfinance sometimes returns MultiIndex columns when downloading
            # multiple tickers. For single ticker, we need to flatten this.
            # MultiIndex looks like: ('Close', 'AAPL') instead of just 'Close'

            if isinstance(df.columns, pd.MultiIndex):
                # get_level_values(0) extracts just the first level ('Close', 'Open', etc.)
                df.columns = df.columns.get_level_values(0)

            #------------------------------------------------------------------
            # STANDARDIZE COLUMN NAMES
            #------------------------------------------------------------------
            # yfinance returns: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
            # We want lowercase with no spaces: 'open', 'high', 'low', 'close', 'adjclose', 'volume'
            # This makes the code more consistent and easier to work with

            column_mapping = {}
            for col in df.columns:
                # Convert to lowercase and remove spaces
                # 'Adj Close' becomes 'adjclose'
                col_lower = str(col).lower().replace(' ', '')
                column_mapping[col] = col_lower

            df.rename(columns=column_mapping, inplace=True)

            #------------------------------------------------------------------
            # HANDLE MISSING 'adjclose' COLUMN
            #------------------------------------------------------------------
            # Some data sources may not have adjusted close price
            # In that case, use regular close price as a fallback

            if 'adjclose' not in df.columns and 'close' in df.columns:
                df['adjclose'] = df['close']

            # Set the index name for clarity
            df.index.name = 'date'

            #------------------------------------------------------------------
            # NEW: SAVE TO LOCAL STORAGE (Requirement d)
            #------------------------------------------------------------------
            if save_local:
                # Create the directory if it doesn't exist
                # os.makedirs with exist_ok=True won't raise error if dir exists
                os.makedirs(local_path, exist_ok=True)
                df.to_csv(local_file)
                print(f"Data saved to: {local_file}")

    elif isinstance(ticker, pd.DataFrame):
        # ticker is already a DataFrame - use it directly
        # This allows users to pass pre-processed data
        df = ticker
    else:
        # Invalid input type
        raise TypeError("ticker can be either a str or a pd.DataFrame instance")

    #--------------------------------------------------------------------------
    # STEP 2: NaN HANDLING (Requirement b)
    #--------------------------------------------------------------------------
    # Check for NaN values before processing
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the data")
        # Option 1: Drop rows with NaN (simplest approach)
        # Option 2: Forward fill (ffill) - use previous value
        # Option 3: Backward fill (bfill) - use next value
        # Option 4: Interpolation - estimate based on surrounding values

        # Using forward fill followed by backward fill handles edge cases
        df.fillna(method='ffill', inplace=True)  # Fill NaN with previous value
        df.fillna(method='bfill', inplace=True)  # Fill remaining NaN with next value
        print(f"NaN values have been handled using forward/backward fill")

    #--------------------------------------------------------------------------
    # INITIALIZE RESULT DICTIONARY
    #--------------------------------------------------------------------------
    # This dictionary will store all outputs from this function
    result = {}

    # Store the original dataframe (before scaling) for reference
    result['df'] = df.copy()

    #--------------------------------------------------------------------------
    # STEP 3: VALIDATE FEATURE COLUMNS
    #--------------------------------------------------------------------------
    # Ensure all requested feature columns exist in the dataframe
    for col in feature_columns:
        # assert raises AssertionError if condition is False
        assert col in df.columns, f"'{col}' does not exist in the dataframe. Available columns: {list(df.columns)}"

    #--------------------------------------------------------------------------
    # STEP 4: ADD DATE COLUMN
    #--------------------------------------------------------------------------
    # If 'date' is not a column (it's the index), add it as a column
    # This is needed later for tracking which dates correspond to which data points
    if "date" not in df.columns:
        df["date"] = df.index

    #--------------------------------------------------------------------------
    # STEP 5: FEATURE SCALING (Requirement e)
    #--------------------------------------------------------------------------
    # MinMaxScaler transforms features to a range [0, 1]
    # Formula: X_scaled = (X - X_min) / (X_max - X_min)
    #
    # Why scale?
    # 1. Neural networks converge faster with normalized inputs
    # 2. Features with larger magnitudes won't dominate the learning
    # 3. Gradient descent works more efficiently
    #
    # IMPORTANT: We store each scaler to inverse transform predictions later

    if scale:
        column_scaler = {}  # Dictionary to store scalers (Requirement e)

        for column in feature_columns:
            # Create a new scaler for each feature
            scaler = preprocessing.MinMaxScaler()

            # fit_transform does two things:
            # 1. fit: learn the min and max values from the data
            # 2. transform: apply the scaling formula
            #
            # np.expand_dims adds a dimension: [1,2,3] -> [[1],[2],[3]]
            # This is required because MinMaxScaler expects 2D input
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))

            # Store the scaler so we can inverse_transform predictions later
            column_scaler[column] = scaler

        # Add scalers to result dictionary for future access
        result["column_scaler"] = column_scaler

    #--------------------------------------------------------------------------
    # STEP 6: CREATE TARGET COLUMN (LABEL)
    #--------------------------------------------------------------------------
    # The 'future' column contains the value we want to predict
    # shift(-lookup_step) moves values UP by lookup_step positions
    #
    # Example with lookup_step=1:
    # adjclose: [100, 101, 102, 103, 104]
    # future:   [101, 102, 103, 104, NaN]  <- shifted up by 1
    #
    # This means: given today's features, predict tomorrow's price

    df['future'] = df['adjclose'].shift(-lookup_step)

    #--------------------------------------------------------------------------
    # STEP 7: SAVE LAST SEQUENCE FOR FUTURE PREDICTION
    #--------------------------------------------------------------------------
    # The last 'lookup_step' rows will have NaN in 'future' column
    # We save these before dropping NaN because they're needed for
    # predicting the actual future (beyond our dataset)

    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    #--------------------------------------------------------------------------
    # STEP 8: DROP NaN VALUES
    #--------------------------------------------------------------------------
    # Remove rows where 'future' is NaN (the last lookup_step rows)
    # These rows don't have labels, so they can't be used for training

    df.dropna(inplace=True)

    #--------------------------------------------------------------------------
    # STEP 9: CREATE SEQUENCES FOR TIME SERIES
    #--------------------------------------------------------------------------
    # LSTM models need sequential data in a specific format
    # We create sequences of n_steps consecutive time points
    #
    # Example with n_steps=3:
    # Sequence 1: [day1, day2, day3] -> predict day4
    # Sequence 2: [day2, day3, day4] -> predict day5
    # etc.
    #
    # deque with maxlen automatically removes oldest item when full
    # This is efficient for creating sliding windows

    sequence_data = []

    # deque is a double-ended queue from collections module
    # maxlen=n_steps means it only keeps the last n_steps items
    # When you append to a full deque, the oldest item is automatically removed
    sequences = deque(maxlen=n_steps)

    # zip combines two iterables element-by-element
    # We iterate through features+date and corresponding future values together
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        # Add current entry to the sequence window
        sequences.append(entry)

        # Only create a training sample when we have enough history
        if len(sequences) == n_steps:
            # np.array(sequences) creates a copy of the current window
            # [sequence_array, target_value] is one training sample
            sequence_data.append([np.array(sequences), target])

    #--------------------------------------------------------------------------
    # STEP 10: PREPARE LAST SEQUENCE FOR FUTURE PREDICTION
    #--------------------------------------------------------------------------
    # Combine the last n_steps from sequences with the saved last_sequence
    # This gives us the most recent data for predicting the actual future
    #
    # s[:len(feature_columns)] removes the date column from each entry

    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    #--------------------------------------------------------------------------
    # STEP 11: SEPARATE FEATURES (X) AND LABELS (y)
    #--------------------------------------------------------------------------
    # X contains the input sequences (features over time)
    # y contains the target values (future prices to predict)

    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # Convert lists to numpy arrays for efficient computation
    X = np.array(X)
    y = np.array(y)

    #--------------------------------------------------------------------------
    # STEP 12: TRAIN/TEST SPLIT (Requirement c)
    #--------------------------------------------------------------------------
    # Two methods available:
    # 1. split_by_date=True: Chronological split (more realistic for time series)
    # 2. split_by_date=False: Random split (standard ML approach)

    if split_by_date:
        # CHRONOLOGICAL SPLIT
        # First (1-test_size)% of data for training
        # Last test_size% of data for testing
        # This mimics real-world scenario: train on past, test on future

        train_samples = int((1 - test_size) * len(X))

        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]

        if shuffle:
            # Shuffle training data to prevent learning temporal patterns
            # Note: We shuffle train and test separately to maintain
            # the temporal integrity within test set
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # RANDOM SPLIT
        # Use sklearn's train_test_split for random splitting
        # This is standard for non-time-series data but can cause
        # "future leakage" in time series (training on future data)

        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = \
            train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    #--------------------------------------------------------------------------
    # STEP 13: EXTRACT TEST DATES AND CREATE TEST DATAFRAME
    #--------------------------------------------------------------------------
    # Get the dates from the test set for later analysis
    # The date is stored in the last column of each sequence's last time step
    # X_test shape: (samples, n_steps, features+1) where +1 is the date

    dates = result["X_test"][:, -1, -1]  # Last column of last time step

    # Retrieve the corresponding rows from original dataframe
    result["test_df"] = result["df"].loc[dates]

    # Remove any duplicated dates (can occur due to indexing)
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

    #--------------------------------------------------------------------------
    # STEP 14: REMOVE DATE COLUMN AND CONVERT TO FLOAT32
    #--------------------------------------------------------------------------
    # Remove the date column from features (it was only for tracking)
    # Convert to float32 for TensorFlow compatibility (saves memory)
    #
    # Slicing: [:, :, :len(feature_columns)]
    # - : = all samples
    # - : = all time steps
    # - :len(feature_columns) = only feature columns (exclude date)

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

    Parameters:
    -----------
    sequence_length : int
        Number of time steps in input sequence (same as n_steps in load_data)

    n_features : int
        Number of features per time step

    units : int, default=256
        Number of LSTM units (neurons) in each layer
        More units = more capacity to learn complex patterns
        But also more prone to overfitting and slower training

    cell : keras layer, default=LSTM
        Type of recurrent cell to use (LSTM, GRU, SimpleRNN)
        LSTM is good for long sequences due to memory cell
        GRU is simpler and often works just as well

    n_layers : int, default=2
        Number of recurrent layers to stack
        More layers = deeper network, can learn more abstract features

    dropout : float, default=0.3
        Dropout rate for regularization (0.3 = drop 30% of connections)
        Helps prevent overfitting by randomly disabling neurons during training

    loss : str, default="mean_absolute_error"
        Loss function to optimize
        MAE: average absolute difference between predicted and actual
        MSE: average squared difference (penalizes large errors more)

    optimizer : str, default="rmsprop"
        Optimization algorithm
        'rmsprop': good for RNNs, adapts learning rate
        'adam': popular choice, combines momentum and RMSprop

    bidirectional : bool, default=False
        If True, process sequences in both directions
        Can capture patterns that depend on future context
        Note: May not be ideal for prediction tasks

    Returns:
    --------
    model : Sequential
        Compiled Keras model ready for training
    """

    model = Sequential()

    # Build the layers dynamically based on n_layers
    for i in range(n_layers):
        if i == 0:
            # FIRST LAYER: Must specify input_shape
            # input_shape = (sequence_length, n_features)
            # This tells the model what shape of data to expect

            if bidirectional:
                # Bidirectional wrapper processes sequence forwards and backwards
                # Output has 2x units because it concatenates both directions
                model.add(Bidirectional(
                    cell(units, return_sequences=True),
                    input_shape=(sequence_length, n_features)
                ))
            else:
                # return_sequences=True: output at each time step (not just last)
                # Needed when stacking multiple LSTM layers
                model.add(cell(units, return_sequences=True,
                             input_shape=(sequence_length, n_features)))

        elif i == n_layers - 1:
            # LAST RECURRENT LAYER: return_sequences=False
            # Only output the final time step's hidden state
            # This single vector goes to the Dense layer for prediction

            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # HIDDEN LAYERS: return_sequences=True
            # Intermediate layers still need to pass sequences to next layer

            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))

        # Add dropout after each recurrent layer
        # Dropout randomly sets a fraction of inputs to 0 during training
        # This prevents the network from relying too heavily on specific neurons
        model.add(Dropout(dropout))

    # OUTPUT LAYER
    # Dense layer with 1 unit for single value prediction (the price)
    # activation="linear" means no activation (f(x) = x)
    # Linear is appropriate for regression (predicting continuous values)
    model.add(Dense(1, activation="linear"))

    # COMPILE THE MODEL
    # This configures the model for training
    # - loss: function to minimize
    # - metrics: values to monitor during training
    # - optimizer: algorithm for updating weights
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model


#------------------------------------------------------------------------------
# MAIN EXECUTION
#------------------------------------------------------------------------------
if __name__ == "__main__":

    #--------------------------------------------------------------------------
    # CONFIGURATION PARAMETERS
    #--------------------------------------------------------------------------
    # Stock to analyze
    COMPANY = "CBA.AX"

    # Date range for data (Requirement a)
    START_DATE = "2020-01-01"
    END_DATE = "2024-07-01"

    # Model parameters
    N_STEPS = 50              # Number of days to look back
    LOOKUP_STEP = 1           # Predict 1 day ahead
    TEST_SIZE = 0.2           # 20% for testing

    # Feature columns to use
    FEATURE_COLUMNS = ['adjclose', 'volume', 'open', 'high', 'low']

    # Model architecture
    N_LAYERS = 2
    UNITS = 256
    DROPOUT = 0.3
    BIDIRECTIONAL = False

    # Training parameters
    EPOCHS = 25
    BATCH_SIZE = 32

    #--------------------------------------------------------------------------
    # LOAD AND PROCESS DATA
    #--------------------------------------------------------------------------
    print("="*60)
    print("STOCK PREDICTION v0.2 - Enhanced Data Processing")
    print("="*60)
    print(f"\nLoading data for {COMPANY}...")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Sequence length: {N_STEPS} days")
    print(f"Prediction horizon: {LOOKUP_STEP} day(s) ahead")
    print(f"Test size: {TEST_SIZE*100}%")
    print(f"Split method: {'By Date' if True else 'Random'}")

    # Call the load_data function with all parameters
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
        save_local=True,        # Save data locally
        load_local=True,        # Try to load from local first
        local_path='data'       # Directory for local storage
    )

    # Display data information
    print(f"\n{'='*60}")
    print("DATA LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total samples in original data: {len(data['df'])}")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Testing samples: {len(data['X_test'])}")
    print(f"Feature shape: {data['X_train'].shape}")
    print(f"  - Samples: {data['X_train'].shape[0]}")
    print(f"  - Time steps: {data['X_train'].shape[1]}")
    print(f"  - Features: {data['X_train'].shape[2]}")

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

    # Print model summary
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

    # Make predictions on test data
    y_pred = model.predict(data["X_test"])

    # Get the actual values (already scaled)
    y_test = data["y_test"]

    # Inverse transform to get actual prices
    # We use the 'adjclose' scaler since that's what we're predicting
    y_pred_original = data["column_scaler"]["adjclose"].inverse_transform(y_pred)
    y_test_original = data["column_scaler"]["adjclose"].inverse_transform(y_test.reshape(-1, 1))

    #--------------------------------------------------------------------------
    # CALCULATE PERFORMANCE METRICS
    #--------------------------------------------------------------------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Flatten arrays for metric calculation
    y_test_flat = y_test_original.flatten()
    y_pred_flat = y_pred_original.flatten()

    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, y_pred_flat)
    mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100

    print(f"\nPERFORMANCE METRICS - v0.2")
    print("="*60)
    print(f"Mean Absolute Error (MAE):      ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Squared Error (MSE):       {mse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R² Score:                       {r2:.4f}")
    print("="*60)

    #--------------------------------------------------------------------------
    # PLOT RESULTS
    #--------------------------------------------------------------------------
    # Plot 1: Training History
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Actual vs Predicted
    plt.subplot(1, 2, 2)
    plt.plot(y_test_flat, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(y_pred_flat, label='Predicted Price', color='red', alpha=0.7)
    plt.title(f'{COMPANY} Stock Price Prediction\nMAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}')
    plt.xlabel('Time (Trading Days)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('v02_results.png', dpi=150)
    print(f"\nResults saved to 'v02_results.png'")
    plt.show()

    #--------------------------------------------------------------------------
    # PREDICT NEXT DAY
    #--------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FUTURE PREDICTION")
    print(f"{'='*60}")

    # Get the last sequence and reshape for prediction
    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    # Predict
    prediction = model.predict(last_sequence)
    predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]

    print(f"Predicted price for next trading day: ${predicted_price:.2f}")
    print("="*60)
