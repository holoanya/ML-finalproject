# Airbnb Nightly-Price Prediction — Berlin

SoSe 2026 Machine Learning final project. Predict the nightly price of Airbnb
listings in **Berlin** from multiple data modalities.

- **City / snapshot:** Berlin, Inside Airbnb scrape **2025-09-23** (pinned).
- **Modalities:** tabular + spatial (coordinates). *(text is planned as a later addition.)*
- **Target:** `log1p(price)` (price parsed from the `"$1,234.00"` string; values are EUR).
- **Models:** Random Forest baseline → (later) Ridge / Gradient Boosting comparison.
- **Evaluation:** frozen train/test split **grouped by `host_id`**, metrics reported in EUR (RMSE/MAE/R²).

## Data facts (Berlin snapshot 2025-09-23)
- 14,274 listings × 79 columns; **price missing for 35.1%** → 9,264 usable rows.
- Label = `listings.price` snapshot, rows with blank price dropped, kept €10–1000.
- `bathrooms` is 34.6% null → parse `bathrooms_text` instead.

## Reproduce
```bash
pip install -r requirements.txt
python src/data/download.py        # downloads the pinned Berlin snapshot into data/raw/
python src/models/rf_baseline.py   # builds features, fits RF, prints held-out metrics
```

## Layout
```
src/
  data/      download.py    (fetch the pinned snapshot into data/raw/)
             prepare.py     (parse price, label = log1p, frozen host-grouped split)
  features/  features.py    (tabular + spatial FeatureBuilder, fit on train only)
  models/    rf_baseline.py (Random Forest baseline + trivial references)
  eval/      metrics.py     (EUR / log RMSE-MAE-R²)
results/     saved metrics + feature importances
```

## Team
Efoghe · Hausmann · Zhou
