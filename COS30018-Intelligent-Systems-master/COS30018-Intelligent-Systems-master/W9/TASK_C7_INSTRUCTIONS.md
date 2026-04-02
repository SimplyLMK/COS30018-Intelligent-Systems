# TASK C.7 INSTRUCTIONS — Sentiment-Based Stock Price Movement Prediction

## What this task is

Task C.7 of COS30018 Intelligent Systems (Option C). This is the **final task** (worth 30 marks). The goal: build a **classification** system that predicts whether CBA.AX stock price will rise or fall on the next trading day, incorporating sentiment analysis from financial news.

**This is fundamentally different from C.1–C.6.** Previous tasks were regression (predict the next price). C.7 is binary classification (predict UP or DOWN), and it requires external data (news sentiment).

## Dependencies to install

```bash
pip install pygooglenews nltk transformers torch yfinance pandas numpy matplotlib seaborn scikit-learn python-dateutil
```

**Notes:**
- `torch` + `transformers` are large (~2GB total). They are needed for FinBERT. First run will also download the FinBERT model (~400MB).
- If `torch` installation fails, try: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `pygooglenews` depends on `feedparser` and `beautifulsoup4` (installed automatically).
- If `pygooglenews` gives import errors, try: `pip install pygooglenews==0.1.2`

## How to run

```bash
cd /path/to/project
python stock_prediction_v07.py
```

**Expected runtime:** 10–30 minutes depending on:
- News collection: 5–15 min (first run only; cached afterwards)
- FinBERT scoring: 2–10 min depending on number of headlines
- Classification experiments: <1 min each

**After first run,** subsequent runs use cached news data and are much faster (~5 min total).

## Expected outputs

### Files generated:
| File | Description |
|------|-------------|
| `data/cba_news_headlines.csv` | Cached news headlines with dates |
| `data/cba_daily_sentiment.csv` | Daily aggregated sentiment scores |
| `experiment_results_v07.csv` | All experiment metrics |
| `plot_c7_sentiment_vs_price.png` | Dual-axis chart: price + sentiment |
| `plot_c7_experiment_comparison.png` | Bar chart comparing all experiments |
| `plot_c7_cm_baseline.png` | Confusion matrix for baseline model |
| `plot_c7_cm_sentiment.png` | Confusion matrix for best sentiment model |
| `plot_c7_feature_importance.png` | Feature importance for best model |
| `plot_c7_roc_curves.png` | ROC curves for all experiments |

### Console output:
- News collection progress
- VADER and FinBERT scoring summaries
- Per-experiment results (accuracy, precision, recall, F1, ROC-AUC)
- Final summary table comparing all experiments
- Sentiment improvement over baseline

## Known issues and troubleshooting

### pygooglenews returns very few headlines
**Expected.** Google News RSS has limited historical coverage. For CBA.AX from 2020, you may get anywhere from 50 to 500 headlines. This is a legitimate finding to discuss in the report.

**If it returns 0 headlines:**
1. Check your internet connection
2. Try running the collection again (Google may have rate-limited you)
3. If persistent, edit `COMPANY_SEARCH_TERMS` in the code to try different queries:
   ```python
   COMPANY_SEARCH_TERMS = [
       "Commonwealth Bank",
       "CBA stock",
       "CommBank",
       "CBA Australia banking"
   ]
   ```
4. As a last resort, the code handles zero headlines gracefully — all experiments will still run but sentiment features will be zero (essentially a baseline-only comparison).

### FinBERT fails to load
If `transformers` or `torch` installation is problematic:
1. The code handles this gracefully — it prints a warning and uses zero-filled FinBERT scores
2. VADER experiments will still run normally
3. You can still demonstrate the methodology and discuss FinBERT conceptually in the report

### Class imbalance
Stock movements tend to be roughly 50/50 over long periods, but in trending markets you may see imbalance (e.g., 60% UP in a bull market). The code uses `class_weight='balanced'` in RandomForest to handle this. If the test set is heavily skewed, note this in the report.

### Low accuracy (50-55%)
**This is expected and normal for stock prediction.** A random baseline is 50%. Getting 55% consistently is actually meaningful in finance. The key assessment criterion is whether sentiment **improves** over the baseline, not whether it achieves high absolute accuracy.

## Collecting results for the report

After running, you need to provide these to the report generation step:

1. **Console output** — copy the full terminal output (especially experiment summaries)
2. **`experiment_results_v07.csv`** — the metrics CSV
3. **All 5+ PNG files** — the plots
4. **`data/cba_news_headlines.csv`** — for reporting data coverage statistics
5. **`data/cba_daily_sentiment.csv`** — for reporting sentiment statistics

Screenshot the console output or redirect to file:
```bash
python stock_prediction_v07.py 2>&1 | tee output_c7.txt
```

---

## Report structure (for DOCX generation)

### Document setup (same as C.4/C.5/C.6 reports)

**Page setup:** A4, 1-inch margins.
**Styles:** Same as previous reports (Arial 12pt body, Heading1 blue 16pt, Heading2 blue 13pt).
**Header:** `"COS30018 — Task C.7 Report"` left, `"Le Minh Kha"` right.
**Footer:** Centered page number.

### Section 1: Introduction

Context: This task extends the stock prediction project from regression (predicting price) to classification (predicting direction). The rationale is that in practice, knowing whether a stock will go up or down is more actionable than predicting an exact price. Incorporating external information (news sentiment) tests whether alternative data sources can improve prediction beyond technical analysis alone.

State the research question: Does incorporating news sentiment improve next-day stock movement prediction compared to using technical indicators alone?

### Section 2: Methodology

**2.1 Data Sources**

- Stock data: CBA.AX from Yahoo Finance, 2020-01-01 to 2024-07-01. Same dataset as previous tasks.
- News data: Google News headlines collected via pygooglenews. Report actual numbers: total headlines, unique headlines, date coverage, average headlines per trading day.

Code snippet: `collect_news_pygooglenews()` function — show the month-by-month chunking approach and caching mechanism.

**2.2 Sentiment Analysis**

Two sentiment tools compared:

*VADER:* Rule-based model using a lexicon of ~7500 words with pre-assigned valence scores. Designed for social media text. Returns compound score in [-1, +1].

Code snippet: Show `score_vader()` call.

*FinBERT:* BERT transformer fine-tuned on 10,000+ financial texts (financial news, analyst reports). Classifies into positive/negative/neutral. Returns confidence score converted to [-1, +1].

Code snippet: Show `score_finbert()` call.

Key difference: VADER is domain-agnostic; FinBERT understands financial language. For example, "bank downgrades outlook" — VADER may score "downgrades" mildly negative, while FinBERT correctly identifies this as strongly negative in financial context.

**2.3 Technical Indicators (Features)**

List all features with brief explanations:
- Daily returns and 5 lagged returns
- RSI (14-day): momentum oscillator
- MACD, signal line, histogram: trend-following momentum
- Bollinger %B: volatility-based position
- Volume change: trading activity momentum
- Price/SMA ratios: trend deviation
- Volatility (5-day rolling std)

Code snippet: `compute_rsi()` and `compute_macd()` implementations.

**2.4 Classification Target**

Binary label: 1 if next day's adjclose > today's adjclose, else 0.

Code snippet: `create_binary_target()` function.

**2.5 Experiment Design**

Six experiments:
1. Baseline RF (technical features only) — 15 features
2. Tech + VADER (RF) — 21 features
3. Tech + FinBERT (RF) — 18 features
4. Tech + VADER + FinBERT (RF) — 24 features
5. Baseline GB (technical features only) — 15 features
6. Tech + VADER + FinBERT (GB) — 24 features

All use chronological 80/20 split. StandardScaler fitted on training set. RandomForest: 200 trees, max_depth=10, balanced class weights. GradientBoosting: 200 trees, max_depth=5, learning_rate=0.05.

Page break after methodology.

### Section 3: Data Collection Results

**3.1 News Coverage**

Report statistics from `cba_news_headlines.csv`:
- Total headlines collected
- Unique headlines after deduplication
- Date range covered
- Trading days with at least one headline vs total trading days (% coverage)
- Average headlines per covered day

Table: Monthly headline counts.

Discuss limitations: Google News RSS coverage gaps, potential bias toward recent events, geographic filtering.

**3.2 Sentiment Score Distributions**

Figure 1: Histogram of VADER compound scores and FinBERT scores side by side.
Figure 2: `plot_c7_sentiment_vs_price.png` — sentiment vs price timeline.

Discuss: How do VADER and FinBERT scores compare? Do they agree? Are there periods where they diverge? Does sentiment appear to track price movements?

Page break.

### Section 4: Classification Results

**Table 1:** Full results table — all experiments.
Columns: Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC, N Features, Time (s).
Use actual numbers from `experiment_results_v07.csv`.
Column widths: `[2800, 900, 900, 800, 900, 900, 800, 700]`

**4.1 Baseline Performance**

Discuss the technical-only baseline. Compare accuracy to random (50%). What does the confusion matrix tell us about the model's bias (does it favour predicting UP or DOWN)?

Figure 3: `plot_c7_cm_baseline.png` — baseline confusion matrix.

**4.2 Effect of Sentiment Features**

Compare sentiment-enhanced models against baseline. Did VADER improve accuracy? Did FinBERT? Did combining both help more than either alone?

Figure 4: `plot_c7_cm_sentiment.png` — best sentiment model confusion matrix.

**4.3 VADER vs FinBERT (Independent Research)**

This is the key comparison. Discuss:
- Which sentiment tool produced better classification performance?
- Is the difference statistically meaningful or within noise?
- Which model's feature importance ranking placed sentiment features higher?
- Does financial-domain specificity of FinBERT translate to better predictive power?

Figure 5: `plot_c7_experiment_comparison.png` — bar chart comparison.

**4.4 Feature Importance**

Discuss which features the model relied on most. Did technical indicators dominate, or did sentiment features contribute meaningfully?

Figure 6: `plot_c7_feature_importance.png` — feature importance bar chart.

**4.5 ROC Curve Analysis**

Figure 7: `plot_c7_roc_curves.png` — ROC curves for all models.

Discuss: A model with AUC > 0.5 is better than random. Compare AUC values across experiments. A higher AUC from the sentiment model would indicate that sentiment provides useful discriminative signal even if raw accuracy improvement is small.

Page break.

### Section 5: Discussion

**5.1 Why Stock Direction Prediction is Hard**

Stock prices are influenced by countless factors beyond news headlines: macroeconomic conditions, interest rates, institutional trading, global events. Efficient Market Hypothesis (weak form) suggests that past prices already contain historical information. Our technical indicators capture price patterns, while sentiment captures market mood — but neither captures everything.

Expected accuracy of 50-55% is consistent with academic literature on short-term stock prediction using sentiment.

**5.2 Data Quality and Coverage**

Google News is not a comprehensive source. Bloomberg Terminal, Refinitiv, or Twitter firehose would provide better coverage but are not freely accessible. The sparse coverage means our sentiment features are often forward-filled (carrying stale information), which reduces their predictive value. Better data would likely improve results.

**5.3 VADER vs FinBERT: Domain Matters**

[Discuss based on actual results]. If FinBERT outperformed VADER: domain-specific models capture financial nuance that generic sentiment tools miss. If VADER performed comparably: for headline-level sentiment (short text), VADER's rule-based approach is sufficient, and FinBERT's strength may be more apparent on longer financial documents.

**5.4 Regression vs Classification**

Compare philosophically with the regression approach from C.1–C.6. Regression models achieved R² > 0.9 (seemingly excellent), but much of that comes from price autocorrelation (today's price ≈ yesterday's price). Classification strips away this illusion by asking the harder question: will the price go UP or DOWN? The lower accuracy here is actually a more honest measure of predictive ability.

**5.5 Limitations and Future Work**

- More comprehensive news sources (Bloomberg, Twitter/X)
- Intraday sentiment (hourly or minute-level)
- Different classification targets (e.g., predicting magnitude of change, or multi-class: strong up / mild up / down)
- LSTM-based classifier using sequential sentiment
- Alternative features: earnings call transcripts, insider trading signals

Page break.

### Section 6: Conclusion

Summarise: Built a classification pipeline combining technical indicators with news sentiment to predict next-day stock movement. Tested six experiment configurations across two classifier types. Report the key finding: did sentiment improve prediction? By how much? Which sentiment tool performed better?

State that while absolute accuracy is modest (as expected for stock prediction), the methodology demonstrates a complete pipeline from data collection through sentiment scoring to model evaluation, and the comparison between VADER and FinBERT provides insight into the value of domain-specific NLP tools.

### References

- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. arXiv:1908.10063.
- scikit-learn documentation. https://scikit-learn.org/stable/
- pygooglenews. https://pypi.org/project/pygooglenews/
- Yahoo Finance. CBA.AX historical data. https://finance.yahoo.com/quote/CBA.AX/
- Fama, E.F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance.
- Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. Journal of Computational Science.

---

## File locations

| File | Path |
|------|------|
| v07 code | `stock_prediction_v07.py` |
| DOCX skill | `/mnt/skills/public/docx/SKILL.md` |
| News cache | `data/cba_news_headlines.csv` (auto-generated) |
| Sentiment cache | `data/cba_daily_sentiment.csv` (auto-generated) |
| Results CSV | `experiment_results_v07.csv` (auto-generated) |
| Plot images | `plot_c7_*.png` (auto-generated) |

---

## Writing style

Cal Newport's style: direct, no fluff, no em dashes, structured reasoning with concrete examples. Academic tone but not overly formal. Every claim should have a number behind it.

---

## Workflow summary

1. **MK installs dependencies** (especially `transformers`, `torch`, `pygooglenews`)
2. **MK runs `stock_prediction_v07.py`** — first run collects news and takes longer; subsequent runs use cache
3. **MK collects outputs:** console output, CSV, all PNG plots
4. **MK provides those artifacts** to the next Claude session
5. **Claude reads this instruction file + DOCX skill**, generates `generate_c7_report.js`
6. **Run, validate, copy report to `/mnt/user-data/outputs/`**

If this handoff goes to Claude Code, the agent should ask MK for the experiment results before writing the report. Without actual numbers, the report cannot be generated.

---

## Status of C.6

C.6 is partially done. The v06 code exists but hasn't been fully run. MK should:
1. Finish running v06 and collecting results
2. Generate the C.6 report
3. Then run v07 for C.7

C.7 is the final task and carries 30 marks. Prioritise getting v07 running and producing results.
