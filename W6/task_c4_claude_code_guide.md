# Task C.4 — Claude Code Guide

## PART 1: INPUT FILES YOU NEED TO PROVIDE

Give Claude Code these files:

1. **`stock_prediction_v03.py`** — Your current codebase (this is the starting point)
2. **`TASK_C4.md`** — The instructions below (Part 2)
3. **`Tasks_C_4_-_Machine_Learning_1.pdf`** — The original task sheet (optional but helpful for reference)

That's it. Claude Code doesn't need your old reports, screenshots, or the P1 code — all the relevant context is baked into the instructions below.

---

## PART 2: INSTRUCTIONS FOR CLAUDE CODE

Copy everything below this line and use it as your prompt/instructions file.

---

# Task C.4: Machine Learning 1 — Dynamic Model Builder + Experiments

## Context

You are working on `stock_prediction_v04.py`, which builds on v0.3. The existing codebase already has:
- `load_data()` — flexible data loading with date ranges, NaN handling (ffill/bfill), per-column MinMaxScaler stored in `column_scaler` dict, train/test split options, local CSV caching
- `plot_candlestick()` and `plot_boxplot()` — visualization functions
- A hardcoded LSTM model built manually with `Sequential()` and explicit `model.add()` calls
- Evaluation metrics: MAE, RMSE, MAPE, R²
- Stock: CBA.AX, date range 2020-01-01 to 2024-07-01
- Features: adjclose, volume, open, high, low
- Libraries: TensorFlow/Keras, pandas, numpy, sklearn, yfinance, mplfinance, matplotlib

## What to build (two deliverables)

### Deliverable 1: `create_model()` function

Replace the hardcoded model construction with a flexible function. Signature:

```python
def create_model(input_shape, layer_configs, dropout_rate=0.2, loss="mean_absolute_error", optimizer="adam"):
    """
    Build a Deep Learning model dynamically from a configuration list.
    
    Parameters:
        input_shape: tuple, shape of input data (n_steps, n_features)
        layer_configs: list of dicts, each with:
            - "type": str — one of "LSTM", "GRU", "SimpleRNN", "Dense"
            - "units": int — number of units/neurons
            - "params": dict (optional) — extra kwargs passed to the layer
        dropout_rate: float, dropout applied after each recurrent layer (0 = no dropout)
        loss: str, loss function name
        optimizer: str, optimizer name
    
    Returns:
        compiled tf.keras.Model
    """
```

Example usage:
```python
# 2-layer LSTM with dropout
model = create_model(
    input_shape=(50, 5),
    layer_configs=[
        {"type": "LSTM", "units": 50},
        {"type": "LSTM", "units": 50},
        {"type": "Dense", "units": 1}
    ],
    dropout_rate=0.2,
    loss="mean_absolute_error",
    optimizer="adam"
)
```

**Implementation requirements:**

1. **Layer type mapping** — Use a dict to map string names to Keras layer classes:
   ```python
   LAYER_MAP = {
       "LSTM": tf.keras.layers.LSTM,
       "GRU": tf.keras.layers.GRU,
       "SimpleRNN": tf.keras.layers.SimpleRNN,
       "Dense": tf.keras.layers.Dense
   }
   ```

2. **`return_sequences` logic** — For stacked recurrent layers, all recurrent layers EXCEPT the last one need `return_sequences=True`. The function must figure this out automatically by looking ahead in the config list. Specifically: a recurrent layer needs `return_sequences=True` if any subsequent layer in the config is also a recurrent type (LSTM/GRU/SimpleRNN).

3. **`input_shape`** — Only the first layer gets the `input_shape` parameter.

4. **Dropout** — Add a `tf.keras.layers.Dropout(dropout_rate)` after each recurrent layer (not after Dense layers). Skip if `dropout_rate == 0`.

5. **Final Dense layer** — If the last layer in `layer_configs` is not a Dense layer, automatically append `Dense(1)` for single-value regression output.

6. **Validation** — Raise a `ValueError` if `layer_configs` is empty or if an unknown layer type is provided.

### Deliverable 2: Experiment runner + results

Create a section in the main script (or a separate `run_experiments.py`) that systematically tests different configurations. Structure it as a grid:

**Experiment 1 — Network type comparison** (isolate cell type effect):
| Config | Layers | Units | Epochs | Batch |
|--------|--------|-------|--------|-------|
| LSTM   | 2      | 50    | 25     | 32    |
| GRU    | 2      | 50    | 25     | 32    |
| SimpleRNN | 2   | 50    | 25     | 32    |

**Experiment 2 — Depth comparison** (isolate layer count effect):
| Config | Type | Layers | Units | Epochs | Batch |
|--------|------|--------|-------|--------|-------|
| Shallow | LSTM | 1     | 50    | 25     | 32    |
| Medium  | LSTM | 2     | 50    | 25     | 32    |
| Deep    | LSTM | 3     | 50    | 25     | 32    |

**Experiment 3 — Width comparison** (isolate unit count effect):
| Config | Type | Layers | Units | Epochs | Batch |
|--------|------|--------|-------|--------|-------|
| Narrow | LSTM | 2      | 25    | 25     | 32    |
| Medium | LSTM | 2      | 50    | 25     | 32    |
| Wide   | LSTM | 2      | 100   | 25     | 32    |

**Experiment 4 — Training hyperparameters** (vary epochs + batch size):
| Config | Type | Layers | Units | Epochs | Batch |
|--------|------|--------|-------|--------|-------|
| A      | LSTM | 2      | 50    | 25     | 32    |
| B      | LSTM | 2      | 50    | 50     | 32    |
| C      | LSTM | 2      | 50    | 25     | 16    |
| D      | LSTM | 2      | 50    | 50     | 64    |

**For each experiment, collect and print:**
- MAE, RMSE, MAPE, R²
- Training time (seconds)
- Store results in a list of dicts or pandas DataFrame
- At the end, print a summary comparison table

**Implementation pattern:**
```python
import time

experiments = [
    {"name": "LSTM-2x50", "layers": [{"type": "LSTM", "units": 50}, {"type": "LSTM", "units": 50}], "epochs": 25, "batch_size": 32},
    # ... more configs
]

results = []
for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"{'='*60}")
    
    model = create_model(
        input_shape=(n_steps, n_features),
        layer_configs=exp["layers"] + [{"type": "Dense", "units": 1}],
        dropout_rate=0.2
    )
    
    start_time = time.time()
    model.fit(X_train, y_train, epochs=exp["epochs"], batch_size=exp["batch_size"], validation_split=0.1, verbose=1)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test)
    # ... compute MAE, RMSE, MAPE, R²
    
    results.append({
        "name": exp["name"],
        "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
        "train_time": train_time
    })

# Print summary table
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(results_df.to_string(index=False))
```

## Code style rules

- Keep all existing functions (load_data, plot_candlestick, plot_boxplot) untouched
- Add `create_model()` as a new function near the top of the file (after imports, before load_data)
- Add the experiment code in the `if __name__ == "__main__"` block
- Use the same random seeds as v0.3 for reproducibility (np.random.seed(314), tf.random.set_seed(314), random.seed(314))
- Use the same data loading config as v0.3 (CBA.AX, 2020-01-01 to 2024-07-01, same features)
- Comment non-trivial code — this is an academic project where I need to explain the code in a report
- Print model.summary() for at least the first experiment so I can screenshot it

## Output

The final file should be `stock_prediction_v04.py` — a single Python file that:
1. Contains all v0.3 functionality (load_data, visualizations) unchanged
2. Adds the `create_model()` function
3. Runs the experiment grid in `__main__`
4. Prints a formatted results summary table at the end
