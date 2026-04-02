# File: stock_prediction_v07.py
# Author: Le Minh Kha
# Version: 0.7 - Sentiment-Based Stock Price Movement Prediction
#
# This version implements Task C.7: a classification system that predicts
# whether CBA.AX stock price will rise or fall on the next trading day,
# incorporating sentiment analysis from financial news headlines.
#
# Key components:
# - News headline collection via pygooglenews (cached to CSV)
# - Sentiment scoring with VADER (generic) and FinBERT (financial-domain)
# - Technical indicator computation (RSI, MACD, Bollinger Bands, etc.)
# - Binary classification (Random Forest, Gradient Boosting)
# - Baseline vs sentiment-enhanced model comparison
#
# References:
# - Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based
#   Model for Sentiment Analysis of Social Media Text. ICWSM.
# - Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained
#   Language Models. arXiv:1908.10063.
# - scikit-learn classifiers: https://scikit-learn.org/stable/
# - pygooglenews: https://pypi.org/project/pygooglenews/

#------------------------------------------------------------------------------
# IMPORT LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import time
import warnings
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(314)


#------------------------------------------------------------------------------
# CONFIGURATION
#------------------------------------------------------------------------------
COMPANY = "CBA.AX"
COMPANY_SEARCH_TERMS = [
    "Commonwealth Bank Australia",
    "CBA ASX stock",
    "CommBank financial"
]
START_DATE = "2020-01-01"
END_DATE = "2024-07-01"
TEST_SIZE = 0.2    # chronological split
NEWS_CACHE_PATH = "data/cba_news_headlines.csv"
SENTIMENT_CACHE_PATH = "data/cba_daily_sentiment.csv"
RESULTS_CSV_PATH = "experiment_results_v07.csv"


#==============================================================================
# SECTION 1: STOCK DATA LOADING
#==============================================================================

# def load_stock_data(ticker, start_date, end_date):
#     """
#     Load stock price data from Yahoo Finance.

#     Unlike v06's load_data() which creates windowed sequences for DL models,
#     this function returns a simple DataFrame with daily OHLCV data. The
#     classification pipeline handles feature engineering separately.

#     Parameters:
#         ticker: str, stock ticker symbol (e.g., "CBA.AX")
#         start_date: str, "YYYY-MM-DD"
#         end_date: str, "YYYY-MM-DD"

#     Returns:
#         pd.DataFrame with columns: open, high, low, close, adjclose, volume
#         Index: DatetimeIndex named 'date'
#     """
#     print(f"Downloading stock data for {ticker}...")
#     df = yf.download(ticker, start=start_date, end=end_date,
#                      progress=False, auto_adjust=False)

#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)

#     # Normalise column names
#     col_map = {}
#     for col in df.columns:
#         col_map[col] = str(col).lower().replace(' ', '')
#     df.rename(columns=col_map, inplace=True)

#     if 'adjclose' not in df.columns and 'close' in df.columns:
#         df['adjclose'] = df['close']

#     df.index.name = 'date'

#     # Handle NaN
#     nan_count = df.isna().sum().sum()
#     if nan_count > 0:
#         print(f"  Warning: {nan_count} NaN values found, forward-filling")
#         df.fillna(method='ffill', inplace=True)
#         df.fillna(method='bfill', inplace=True)

#     print(f"  Loaded {len(df)} trading days: "
#           f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
#     return df
def load_stock_data(company, start_date, end_date):
    print(f"Loading offline stock data for {company}...")
    
    # Update this path if your CSV is in a different folder!
    file_path = "CBA.AX_2020-01-01_2024-07-01.csv" 
    
    try:
        # Load the CSV, setting the lowercase 'date' column as the index
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        
        # Rename columns to perfectly match standard yfinance output
        df = df.rename(columns={
            'adjclose': 'Adj Close',
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'open': 'Open',
            'volume': 'Volume'
        })
        
        # Capitalize the index name as well
        df.index.name = 'Date'
        
        # Sort index and filter by your date config
        df = df.sort_index()
        df = df.loc[start_date:end_date]
        
    except FileNotFoundError:
        print(f"Error: Could not find the offline data file at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading offline data: {e}")
        return None

    if not df.empty:
        print(f"Successfully loaded data from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    else:
        print("Warning: The dataframe is empty after filtering by dates.")
        
    return df

#==============================================================================
# SECTION 2: NEWS DATA COLLECTION
#==============================================================================

def collect_news_pygooglenews(search_terms, start_date, end_date,
                              cache_path=NEWS_CACHE_PATH):
    """
    Collect news headlines from Google News using pygooglenews.

    Queries are chunked by month to handle Google News's result limits.
    Results are cached to CSV so collection only needs to run once.

    Google News RSS feeds have limited historical coverage. For dates
    more than ~1 year old, coverage will be sparse. This is a known
    limitation discussed in the report.

    Parameters:
        search_terms: list of str, search queries to try
        start_date: str, "YYYY-MM-DD"
        end_date: str, "YYYY-MM-DD"
        cache_path: str, path to save/load cached headlines

    Returns:
        pd.DataFrame with columns: date, title, source
    """
    # Check cache first
    if os.path.exists(cache_path):
        print(f"  Loading cached news headlines from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['date'])
        print(f"  Loaded {len(df)} cached headlines")
        return df

    print("  Collecting news headlines from Google News...")
    print("  (This may take several minutes due to rate limiting)")

    try:
        from pygooglenews import GoogleNews
    except ImportError:
        print("  ERROR: pygooglenews not installed.")
        print("  Run: pip install pygooglenews")
        print("  Falling back to empty dataset.")
        return pd.DataFrame(columns=['date', 'title', 'source'])

    gn = GoogleNews(lang='en', country='AU')
    all_headlines = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Chunk by month
    current = start
    month_count = 0
    while current < end:
        next_month = current + relativedelta(months=1)
        if next_month > end:
            next_month = end

        from_str = current.strftime("%Y-%m-%d")
        to_str = next_month.strftime("%Y-%m-%d")

        for term in search_terms:
            try:
                result = gn.search(term, from_=from_str, to_=to_str)
                entries = result.get('entries', [])

                for entry in entries:
                    pub_date = entry.get('published', '')
                    title = entry.get('title', '')
                    source = entry.get('source', {})
                    source_name = source.get('title', 'Unknown') if isinstance(source, dict) else str(source)

                    # Parse date
                    try:
                        if pub_date:
                            parsed_date = pd.to_datetime(pub_date, utc=True)
                            parsed_date = parsed_date.tz_localize(None).normalize()
                        else:
                            continue
                    except Exception:
                        continue

                    all_headlines.append({
                        'date': parsed_date,
                        'title': title,
                        'source': source_name
                    })

                # Rate limit: wait between queries to avoid blocking
                time.sleep(1.5)

            except Exception as e:
                print(f"    Warning: Failed to fetch '{term}' "
                      f"for {from_str}: {str(e)[:80]}")
                time.sleep(3)

        month_count += 1
        if month_count % 6 == 0:
            print(f"    Processed {month_count} months, "
                  f"{len(all_headlines)} headlines so far...")
        current = next_month

    df = pd.DataFrame(all_headlines)

    if len(df) > 0:
        # Remove duplicates by title
        df = df.drop_duplicates(subset='title', keep='first')
        df = df.sort_values('date').reset_index(drop=True)

        # Remove headlines that are just source attributions
        # (Google News appends " - SourceName" to titles)
        df['title_clean'] = df['title'].apply(
            lambda x: x.rsplit(' - ', 1)[0] if ' - ' in x else x
        )
    else:
        df = pd.DataFrame(columns=['date', 'title', 'source', 'title_clean'])

    # Cache results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Collected {len(df)} unique headlines, saved to {cache_path}")

    return df


#==============================================================================
# SECTION 3: SENTIMENT ANALYSIS
#==============================================================================

def setup_vader():
    """
    Initialise NLTK VADER sentiment analyser.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
    sentiment tool designed for social media text. It uses a lexicon of words
    with pre-assigned sentiment scores plus rules for handling negation,
    punctuation emphasis, capitalization, etc.

    For financial text, VADER has a known limitation: financial jargon like
    "bearish" or "downgrade" may not be in its lexicon. This is why we
    compare it against FinBERT.

    Reference:
        Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based
        Model for Sentiment Analysis of Social Media Text.
    """
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()


def score_vader(headlines, vader_analyzer):
    """
    Score a list of headlines using VADER.

    Parameters:
        headlines: list of str
        vader_analyzer: SentimentIntensityAnalyzer instance

    Returns:
        list of dicts, each with keys: compound, pos, neg, neu
    """
    scores = []
    for headline in headlines:
        score = vader_analyzer.polarity_scores(str(headline))
        scores.append({
            'vader_compound': score['compound'],
            'vader_pos': score['pos'],
            'vader_neg': score['neg'],
            'vader_neu': score['neu']
        })
    return scores


def setup_finbert():
    """
    Load the ProsusAI/finbert model for financial sentiment analysis.

    FinBERT is a BERT model fine-tuned on financial text (10-K filings,
    analyst reports, financial news). It classifies text into three
    categories: positive, negative, neutral.

    Compared to VADER, FinBERT understands financial context. For example,
    "the company posted a loss" is clearly negative in FinBERT but VADER
    might miss the financial connotation of "loss" vs emotional "loss".

    First run will download the model (~400MB). Subsequent runs use cache.

    Reference:
        Araci, D. (2019). FinBERT: Financial Sentiment Analysis with
        Pre-Trained Language Models. arXiv:1908.10063.

    Returns:
        transformers pipeline object
    """
    try:
        from transformers import pipeline
        print("  Loading FinBERT model (first run downloads ~400MB)...")
        finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1  # CPU; use 0 for GPU
        )
        print("  FinBERT loaded successfully")
        return finbert
    except ImportError:
        print("  ERROR: transformers/torch not installed.")
        print("  Run: pip install transformers torch")
        return None
    except Exception as e:
        print(f"  ERROR loading FinBERT: {e}")
        return None


def score_finbert(headlines, finbert_pipeline, batch_size=32):
    """
    Score headlines using FinBERT.

    FinBERT outputs a label (positive/negative/neutral) and a confidence
    score. We convert this to a single numeric score:
        positive -> +confidence
        negative -> -confidence
        neutral  -> 0.0

    Parameters:
        headlines: list of str
        finbert_pipeline: transformers pipeline
        batch_size: int, process headlines in batches

    Returns:
        list of dicts with: finbert_score, finbert_label, finbert_confidence
    """
    if finbert_pipeline is None:
        return [{'finbert_score': 0.0, 'finbert_label': 'neutral',
                 'finbert_confidence': 0.0}] * len(headlines)

    scores = []
    # Process in batches to manage memory
    for i in range(0, len(headlines), batch_size):
        batch = [str(h)[:512] for h in headlines[i:i + batch_size]]
        try:
            results = finbert_pipeline(batch, truncation=True,
                                        max_length=512)
            for r in results:
                label = r['label'].lower()
                conf = r['score']
                if label == 'positive':
                    numeric = conf
                elif label == 'negative':
                    numeric = -conf
                else:
                    numeric = 0.0

                scores.append({
                    'finbert_score': round(numeric, 4),
                    'finbert_label': label,
                    'finbert_confidence': round(conf, 4)
                })
        except Exception as e:
            print(f"    FinBERT batch error at index {i}: {e}")
            for _ in batch:
                scores.append({
                    'finbert_score': 0.0,
                    'finbert_label': 'neutral',
                    'finbert_confidence': 0.0
                })

        if (i + batch_size) % 100 == 0 and i > 0:
            print(f"    Scored {i + batch_size}/{len(headlines)} headlines")

    return scores


def aggregate_daily_sentiment(news_df, stock_dates):
    """
    Aggregate headline-level sentiment scores to daily trading-day level.

    For each trading day, compute:
    - Mean sentiment scores across all headlines published that day
    - Number of headlines (news volume)
    - Rolling 3-day and 5-day sentiment averages

    Days with no news coverage receive NaN, which is then forward-filled
    (carrying the last known sentiment forward). This is a common approach:
    if there is no new information, the market's sentiment state persists.

    Parameters:
        news_df: DataFrame with date, vader_compound, finbert_score, etc.
        stock_dates: DatetimeIndex of trading days

    Returns:
        pd.DataFrame indexed by trading day with sentiment features
    """
    if len(news_df) == 0:
        print("  WARNING: No news data available. Using neutral sentiment.")
        empty = pd.DataFrame(index=stock_dates)
        for col in ['vader_compound', 'vader_pos', 'vader_neg',
                     'finbert_score', 'news_count',
                     'vader_ma3', 'vader_ma5',
                     'finbert_ma3', 'finbert_ma5']:
            empty[col] = 0.0
        empty['news_count'] = 0
        return empty

    # Ensure date column is datetime with no timezone
    news_df = news_df.copy()
    news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()

    # Group by date and aggregate
    sentiment_cols = [c for c in news_df.columns if c.startswith(('vader_', 'finbert_'))
                      and c not in ('finbert_label', 'finbert_confidence')]

    daily = news_df.groupby('date').agg(
        **{col: (col, 'mean') for col in sentiment_cols if col in news_df.columns},
        **{'news_count': ('title', 'count')}
    )

    # Reindex to trading days
    daily = daily.reindex(stock_dates)

    # Forward-fill: carry last known sentiment for days with no news
    for col in sentiment_cols:
        if col in daily.columns:
            daily[col] = daily[col].fillna(method='ffill').fillna(0.0)
    daily['news_count'] = daily['news_count'].fillna(0).astype(int)

    # Rolling averages for sentiment momentum
    if 'vader_compound' in daily.columns:
        daily['vader_ma3'] = daily['vader_compound'].rolling(3, min_periods=1).mean()
        daily['vader_ma5'] = daily['vader_compound'].rolling(5, min_periods=1).mean()
    if 'finbert_score' in daily.columns:
        daily['finbert_ma3'] = daily['finbert_score'].rolling(3, min_periods=1).mean()
        daily['finbert_ma5'] = daily['finbert_score'].rolling(5, min_periods=1).mean()

    return daily


#==============================================================================
# SECTION 4: TECHNICAL INDICATORS
#==============================================================================

def compute_rsi(prices, period=14):
    """
    Compute Relative Strength Index (RSI).

    RSI measures momentum by comparing the magnitude of recent gains
    to recent losses. Values range from 0 to 100:
    - RSI > 70: potentially overbought (may fall)
    - RSI < 30: potentially oversold (may rise)

    The standard period is 14 trading days.

    Parameters:
        prices: pd.Series of closing prices
        period: int, lookback window (default 14)

    Returns:
        pd.Series of RSI values
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices, fast=12, slow=26, signal=9):
    """
    Compute Moving Average Convergence Divergence (MACD).

    MACD measures the relationship between two exponential moving averages.
    The MACD line is the difference between fast and slow EMAs. The signal
    line is an EMA of the MACD line. The histogram is the difference
    between MACD and signal.

    A positive histogram suggests upward momentum; negative suggests
    downward momentum.

    Parameters:
        prices: pd.Series of closing prices
        fast: int, fast EMA period (default 12)
        slow: int, slow EMA period (default 26)
        signal: int, signal line EMA period (default 9)

    Returns:
        tuple of (macd_line, signal_line, histogram) as pd.Series
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_pctb(prices, period=20, num_std=2):
    """
    Compute Bollinger Band %B (percent B).

    Bollinger Bands are volatility bands placed above and below a moving
    average. %B measures where the current price sits relative to the bands:
    - %B > 1: price is above upper band (overbought signal)
    - %B < 0: price is below lower band (oversold signal)
    - %B = 0.5: price is at the middle band (SMA)

    Using %B instead of raw band values normalises across different
    price levels, making it more useful as a classification feature.

    Parameters:
        prices: pd.Series of closing prices
        period: int, SMA period (default 20)
        num_std: float, number of standard deviations (default 2)

    Returns:
        pd.Series of %B values
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pctb = (prices - lower) / (upper - lower)
    return pctb


def compute_all_technical_features(df):
    """
    Compute a full suite of technical indicators from OHLCV data.

    Features computed:
    - Daily returns (1-day percentage change)
    - Lagged returns (1-5 day lookback)
    - RSI (14-day)
    - MACD line, signal, histogram
    - Bollinger %B (20-day)
    - Volume change (1-day percentage change)
    - Price/SMA ratios (5-day and 20-day)

    Parameters:
        df: pd.DataFrame with columns: adjclose, volume (at minimum)

    Returns:
        pd.DataFrame with technical features appended
    """
    result = df.copy()
    prices = result['adjclose']

    # Daily returns
    result['return_1d'] = prices.pct_change()

    # Lagged returns (previous 1-5 days)
    for lag in range(1, 6):
        result[f'return_lag{lag}'] = result['return_1d'].shift(lag)

    # RSI
    result['rsi_14'] = compute_rsi(prices, period=14)

    # MACD
    macd_line, macd_signal, macd_hist = compute_macd(prices)
    result['macd'] = macd_line
    result['macd_signal'] = macd_signal
    result['macd_hist'] = macd_hist

    # Bollinger %B
    result['bb_pctb'] = compute_bollinger_pctb(prices, period=20)

    # Volume change
    result['volume_change'] = result['volume'].pct_change()

    # Price relative to SMA
    sma5 = prices.rolling(5).mean()
    sma20 = prices.rolling(20).mean()
    result['price_sma5_ratio'] = prices / sma5
    result['price_sma20_ratio'] = prices / sma20

    # Volatility (5-day rolling std of returns)
    result['volatility_5d'] = result['return_1d'].rolling(5).std()

    return result


#==============================================================================
# SECTION 5: FEATURE ENGINEERING
#==============================================================================

def create_binary_target(df, price_col='adjclose'):
    """
    Create binary classification target.

    Target = 1 if next trading day's close > today's close (price goes UP)
    Target = 0 if next trading day's close <= today's close (price goes DOWN)

    Parameters:
        df: pd.DataFrame with price column
        price_col: str, column name for closing price

    Returns:
        pd.Series of binary labels (0 or 1), NaN for last row
    """
    next_close = df[price_col].shift(-1)
    target = (next_close > df[price_col]).astype(int)
    # Last row has no next-day data
    target.iloc[-1] = np.nan
    return target


def build_feature_matrix(stock_df, sentiment_df=None,
                         include_vader=False, include_finbert=False):
    """
    Combine technical indicators and sentiment features into a single
    feature matrix for classification.

    Parameters:
        stock_df: pd.DataFrame with technical indicator columns
        sentiment_df: pd.DataFrame with daily sentiment scores (or None)
        include_vader: bool, whether to include VADER sentiment features
        include_finbert: bool, whether to include FinBERT sentiment features

    Returns:
        tuple of (X, y, feature_names, valid_index)
        X: np.array of shape (n_samples, n_features)
        y: np.array of shape (n_samples,) binary labels
        feature_names: list of str
        valid_index: DatetimeIndex of rows used (after dropping NaN)
    """
    df = stock_df.copy()

    # Technical features
    tech_features = [
        'return_1d', 'return_lag1', 'return_lag2', 'return_lag3',
        'return_lag4', 'return_lag5',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_pctb', 'volume_change',
        'price_sma5_ratio', 'price_sma20_ratio', 'volatility_5d'
    ]

    feature_cols = [f for f in tech_features if f in df.columns]

    # Add sentiment features if requested
    if sentiment_df is not None and include_vader:
        vader_cols = ['vader_compound', 'vader_pos', 'vader_neg',
                      'vader_ma3', 'vader_ma5', 'news_count']
        for col in vader_cols:
            if col in sentiment_df.columns:
                df[col] = sentiment_df[col]
                feature_cols.append(col)

    if sentiment_df is not None and include_finbert:
        finbert_cols = ['finbert_score', 'finbert_ma3', 'finbert_ma5']
        for col in finbert_cols:
            if col in sentiment_df.columns:
                df[col] = sentiment_df[col]
                feature_cols.append(col)

    # Create target
    df['target'] = create_binary_target(df)

    # Drop rows with NaN (from indicator warm-up period and last row)
    subset = df[feature_cols + ['target']].dropna()

    X = subset[feature_cols].values.astype(np.float32)
    y = subset['target'].values.astype(int)
    feature_names = feature_cols
    valid_index = subset.index

    # Replace any remaining inf/-inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names, valid_index


#==============================================================================
# SECTION 6: CLASSIFICATION AND EVALUATION
#==============================================================================

def train_classifier(X_train, y_train, model_type='rf', random_state=314):
    """
    Train a classification model.

    Supports two model types:
    - 'rf': Random Forest Classifier (ensemble of decision trees, majority vote)
    - 'gb': Gradient Boosting Classifier (sequential trees, each correcting
             the previous one's errors)

    Parameters:
        X_train: np.array of shape (n_samples, n_features)
        y_train: np.array of shape (n_samples,)
        model_type: str, 'rf' or 'gb'
        random_state: int

    Returns:
        fitted classifier
    """
    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # handle any class imbalance
        )
    elif model_type == 'gb':
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            subsample=0.8
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test, model_name="Model"):
    """
    Evaluate a classifier with standard metrics.

    Metrics computed:
    - Accuracy: proportion of correct predictions
    - Precision: of predicted positives, how many were actually positive
    - Recall: of actual positives, how many were correctly predicted
    - F1-Score: harmonic mean of precision and recall
    - ROC-AUC: area under the receiver operating characteristic curve
    - Confusion matrix

    For stock movement prediction, a random baseline would achieve ~50%
    accuracy. Anything consistently above 55% is noteworthy.

    Parameters:
        clf: fitted classifier with predict() and predict_proba()
        X_test: np.array
        y_test: np.array
        model_name: str, label for display

    Returns:
        dict with all metrics and the confusion matrix
    """
    y_pred = clf.predict(X_test)

    # Get probability estimates for ROC-AUC
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        roc_auc = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4) if roc_auc is not None else 'N/A',
        'Confusion Matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    print(f"\n  {model_name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"    ROC-AUC:   {roc_auc:.4f}")
    print(f"    Confusion Matrix:\n{cm}")

    return results


def run_experiment(stock_df, sentiment_df, experiment_name,
                   include_vader=False, include_finbert=False,
                   model_type='rf', test_size=TEST_SIZE):
    """
    Run a single classification experiment end-to-end.

    Uses chronological train/test split (no shuffling) to prevent
    lookahead bias. The first (1-test_size) fraction of data is training,
    the rest is test. This mirrors real-world deployment where you train
    on historical data and predict future movements.

    Parameters:
        stock_df: DataFrame with technical features
        sentiment_df: DataFrame with daily sentiment (or None)
        experiment_name: str, label for this experiment
        include_vader: bool
        include_finbert: bool
        model_type: str, 'rf' or 'gb'
        test_size: float, fraction for test set

    Returns:
        dict with evaluation results and trained model
    """
    print(f"\n{'='*50}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*50}")

    # Build features
    X, y, feature_names, valid_index = build_feature_matrix(
        stock_df, sentiment_df,
        include_vader=include_vader,
        include_finbert=include_finbert
    )

    print(f"  Features: {len(feature_names)} ({', '.join(feature_names[:5])}...)")
    print(f"  Samples: {len(X)}")
    print(f"  Class distribution: UP={np.sum(y == 1)}, DOWN={np.sum(y == 0)}")

    # Chronological split (no shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Train class dist: UP={np.sum(y_train == 1)}, "
          f"DOWN={np.sum(y_train == 0)}")
    print(f"  Test class dist: UP={np.sum(y_test == 1)}, "
          f"DOWN={np.sum(y_test == 0)}")

    # Standardise features (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    start_time = time.time()
    clf = train_classifier(X_train_scaled, y_train,
                           model_type=model_type)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    results = evaluate_classifier(clf, X_test_scaled, y_test,
                                   model_name=experiment_name)
    results['Train Time (s)'] = round(train_time, 1)
    results['N Features'] = len(feature_names)
    results['feature_names'] = feature_names
    results['model'] = clf
    results['scaler'] = scaler
    results['y_test'] = y_test

    return results


#==============================================================================
# SECTION 7: VISUALIZATION
#==============================================================================

def plot_confusion_matrix_chart(cm, title, save_path=None):
    """
    Plot confusion matrix as a heatmap.

    Parameters:
        cm: np.array, confusion matrix from sklearn
        title: str
        save_path: str or None
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'],
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_feature_importance(clf, feature_names, title, save_path=None,
                            top_n=15):
    """
    Plot feature importance from a tree-based classifier.

    Parameters:
        clf: fitted classifier with feature_importances_ attribute
        feature_names: list of str
        title: str
        save_path: str or None
        top_n: int, show top N features
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)),
            importances[indices][::-1],
            color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_roc_curves(results_list, save_path=None):
    """
    Plot ROC curves for multiple models on the same chart.

    Parameters:
        results_list: list of dicts from evaluate_classifier,
                      each must have 'y_proba' and the test labels
        save_path: str or None
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, res in enumerate(results_list):
        if res.get('y_proba') is not None and res.get('y_test') is not None:
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_proba'])
            auc_val = res.get('ROC-AUC', 'N/A')
            label = f"{res['Model']} (AUC={auc_val})"
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Model Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_sentiment_vs_price(stock_df, sentiment_df, save_path=None):
    """
    Plot stock price and sentiment scores on a dual-axis chart.

    Parameters:
        stock_df: DataFrame with adjclose
        sentiment_df: DataFrame with vader_compound and/or finbert_score
        save_path: str or None
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(stock_df.index, stock_df['adjclose'],
             'k-', linewidth=1.5, alpha=0.8, label='CBA.AX Price')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($AUD)', fontsize=12, color='black')

    ax2 = ax1.twinx()
    if 'vader_compound' in sentiment_df.columns:
        # Use rolling average for readability
        vader_smooth = sentiment_df['vader_compound'].rolling(10, min_periods=1).mean()
        ax2.plot(sentiment_df.index, vader_smooth,
                 'tab:blue', linewidth=1, alpha=0.6, label='VADER (10d avg)')
    if 'finbert_score' in sentiment_df.columns:
        finbert_smooth = sentiment_df['finbert_score'].rolling(10, min_periods=1).mean()
        ax2.plot(sentiment_df.index, finbert_smooth,
                 'tab:orange', linewidth=1, alpha=0.6, label='FinBERT (10d avg)')

    ax2.set_ylabel('Sentiment Score', fontsize=12, color='tab:blue')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10)

    ax1.set_title('CBA.AX Price vs News Sentiment Over Time',
                  fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_experiment_comparison(results_list, save_path=None):
    """
    Bar chart comparing accuracy and F1-score across experiments.

    Parameters:
        results_list: list of dicts from evaluate_classifier
        save_path: str or None
    """
    models = [r['Model'] for r in results_list]
    accs = [r['Accuracy'] for r in results_list]
    f1s = [r['F1-Score'] for r in results_list]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',
                   color='steelblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1-Score',
                   color='coral', edgecolor='black', alpha=0.8)

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Performance: Baseline vs Sentiment Models',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4,
               label='Random baseline')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


#==============================================================================
# SECTION 8: MAIN EXECUTION — TASK C.7 EXPERIMENTS
#==============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STOCK PREDICTION v0.7 — Task C.7")
    print("Sentiment-Based Stock Price Movement Prediction")
    print("=" * 60)

    # ==================================================================
    # STEP 1: LOAD STOCK DATA
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading stock data")
    print("=" * 60)

    stock_df = load_stock_data(COMPANY, START_DATE, END_DATE)

    # ==================================================================
    # STEP 2: COLLECT NEWS HEADLINES
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Collecting news headlines")
    print("=" * 60)

    news_df = collect_news_pygooglenews(
        COMPANY_SEARCH_TERMS, START_DATE, END_DATE,
        cache_path=NEWS_CACHE_PATH
    )

    # Report coverage statistics
    if len(news_df) > 0:
        news_dates = news_df['date'].dt.normalize().unique()
        trading_days = stock_df.index.normalize().unique()
        coverage = len(set(news_dates) & set(trading_days))
        print(f"  News coverage: {coverage}/{len(trading_days)} trading days "
              f"({100*coverage/len(trading_days):.1f}%)")
    else:
        print("  WARNING: No news headlines collected.")
        print("  The model will run with technical indicators only.")
        print("  Sentiment features will be zero-filled.")

    # ==================================================================
    # STEP 3: SCORE SENTIMENT
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Scoring sentiment")
    print("=" * 60)

    # 3a: VADER
    print("\n--- 3a: VADER Sentiment ---")
    vader = setup_vader()
    if len(news_df) > 0:
        headlines = news_df['title_clean'].tolist() if 'title_clean' in news_df.columns else news_df['title'].tolist()
        vader_scores = score_vader(headlines, vader)
        vader_df = pd.DataFrame(vader_scores)
        for col in vader_df.columns:
            news_df[col] = vader_df[col].values
        print(f"  VADER scored {len(headlines)} headlines")
        print(f"  Mean compound: {news_df['vader_compound'].mean():.4f}")

    # 3b: FinBERT
    print("\n--- 3b: FinBERT Sentiment ---")
    finbert = setup_finbert()
    if len(news_df) > 0 and finbert is not None:
        finbert_scores = score_finbert(headlines, finbert)
        finbert_df = pd.DataFrame(finbert_scores)
        for col in finbert_df.columns:
            news_df[col] = finbert_df[col].values
        print(f"  FinBERT scored {len(headlines)} headlines")
        print(f"  Mean score: {news_df['finbert_score'].mean():.4f}")
    elif finbert is None:
        print("  FinBERT unavailable. Experiments will use VADER only.")
        # Add placeholder columns
        if len(news_df) > 0:
            news_df['finbert_score'] = 0.0
            news_df['finbert_label'] = 'neutral'
            news_df['finbert_confidence'] = 0.0

    # 3c: Aggregate to daily
    print("\n--- 3c: Aggregating to daily sentiment ---")
    daily_sentiment = aggregate_daily_sentiment(news_df, stock_df.index)
    print(f"  Daily sentiment shape: {daily_sentiment.shape}")

    # Save sentiment data
    daily_sentiment.to_csv(SENTIMENT_CACHE_PATH)
    print(f"  Saved to {SENTIMENT_CACHE_PATH}")

    # ==================================================================
    # STEP 4: COMPUTE TECHNICAL FEATURES
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Computing technical indicators")
    print("=" * 60)

    stock_features = compute_all_technical_features(stock_df)
    print(f"  Technical features computed: {stock_features.shape[1]} columns")

    # ==================================================================
    # STEP 5: RUN CLASSIFICATION EXPERIMENTS
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Running classification experiments")
    print("=" * 60)

    experiments_config = [
        {
            'name': 'Baseline (Tech Only, RF)',
            'include_vader': False, 'include_finbert': False,
            'model_type': 'rf'
        },
        {
            'name': 'Tech + VADER (RF)',
            'include_vader': True, 'include_finbert': False,
            'model_type': 'rf'
        },
        {
            'name': 'Tech + FinBERT (RF)',
            'include_vader': False, 'include_finbert': True,
            'model_type': 'rf'
        },
        {
            'name': 'Tech + VADER + FinBERT (RF)',
            'include_vader': True, 'include_finbert': True,
            'model_type': 'rf'
        },
        {
            'name': 'Baseline (Tech Only, GB)',
            'include_vader': False, 'include_finbert': False,
            'model_type': 'gb'
        },
        {
            'name': 'Tech + VADER + FinBERT (GB)',
            'include_vader': True, 'include_finbert': True,
            'model_type': 'gb'
        },
    ]

    all_exp_results = []

    for config in experiments_config:
        results = run_experiment(
            stock_features, daily_sentiment,
            experiment_name=config['name'],
            include_vader=config['include_vader'],
            include_finbert=config['include_finbert'],
            model_type=config['model_type']
        )
        all_exp_results.append(results)

    # ==================================================================
    # STEP 6: PRINT SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    summary_rows = []
    for r in all_exp_results:
        summary_rows.append({
            'Model': r['Model'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1-Score': r['F1-Score'],
            'ROC-AUC': r['ROC-AUC'],
            'N Features': r['N Features'],
            'Time (s)': r['Train Time (s)']
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Best model
    best_idx = summary_df['Accuracy'].idxmax()
    best = summary_df.loc[best_idx]
    print(f"\nBest by accuracy: {best['Model']} "
          f"(Acc={best['Accuracy']}, F1={best['F1-Score']})")

    # ==================================================================
    # STEP 7: GENERATE PLOTS
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Generating plots")
    print("=" * 60)

    # Plot 1: Sentiment vs Price timeline
    plot_sentiment_vs_price(
        stock_df, daily_sentiment,
        save_path="plot_c7_sentiment_vs_price.png"
    )

    # Plot 2: Experiment comparison bar chart
    plot_experiment_comparison(
        all_exp_results,
        save_path="plot_c7_experiment_comparison.png"
    )

    # Plot 3: Confusion matrices for baseline and best sentiment model
    baseline_res = all_exp_results[0]  # Baseline (Tech Only, RF)
    plot_confusion_matrix_chart(
        baseline_res['Confusion Matrix'],
        title=f"Confusion Matrix: {baseline_res['Model']}",
        save_path="plot_c7_cm_baseline.png"
    )

    # Find best sentiment model (RF experiments with sentiment)
    sentiment_models = [r for r in all_exp_results
                        if 'VADER' in r['Model'] or 'FinBERT' in r['Model']]
    if sentiment_models:
        best_sent = max(sentiment_models, key=lambda r: r['Accuracy'])
        plot_confusion_matrix_chart(
            best_sent['Confusion Matrix'],
            title=f"Confusion Matrix: {best_sent['Model']}",
            save_path="plot_c7_cm_sentiment.png"
        )

    # Plot 4: Feature importance for best model
    best_overall = max(all_exp_results, key=lambda r: r['Accuracy'])
    plot_feature_importance(
        best_overall['model'],
        best_overall['feature_names'],
        title=f"Feature Importance: {best_overall['Model']}",
        save_path="plot_c7_feature_importance.png"
    )

    # Plot 5: ROC curves
    plot_roc_curves(
        all_exp_results,
        save_path="plot_c7_roc_curves.png"
    )

    # ==================================================================
    # STEP 8: SAVE ALL RESULTS
    # ==================================================================
    results_for_csv = []
    for r in all_exp_results:
        results_for_csv.append({
            'Model': r['Model'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1-Score': r['F1-Score'],
            'ROC-AUC': r['ROC-AUC'],
            'N Features': r['N Features'],
            'Train Time (s)': r['Train Time (s)']
        })

    results_df = pd.DataFrame(results_for_csv)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nResults saved to {RESULTS_CSV_PATH}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — ALL EXPERIMENTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Compute sentiment improvement
    baseline_acc = all_exp_results[0]['Accuracy']
    for r in all_exp_results[1:]:
        if 'VADER' in r['Model'] or 'FinBERT' in r['Model']:
            diff = r['Accuracy'] - baseline_acc
            sign = '+' if diff >= 0 else ''
            print(f"  {r['Model']}: {sign}{diff:.4f} vs baseline")

    print("\n" + "=" * 60)
    print("Task C.7 complete.")
    print("=" * 60)
