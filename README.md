# Emirates Auction — Web Scraping & Python/WEKA Data Mining

**Portfolio status: original student-supplied scraping code recovered and adapted; validation on real auction data remains open.** This is a reconstruction of the coursework workflow, **not** a claim of 500+ successfully collected records or verified predictive accuracy.

## Origin & scope

The student's July 2025 project conversations contain an actual Python script using Selenium and BeautifulSoup with the `list-card-container` vehicle selector and `title_...` / `CARD_PRICE_...` identifiers. The student reported the script was working in a later conversation; earlier attempts recorded 0 results. The original final CSV and WEKA evaluation outputs were **not** in the ChatGPT export.

The coursework required 500+ web-scraped instances; missing-value handling, scaling, outlier treatment, categorical encoding; 3 classifiers; correlation-based selection and PCA; WEKA filters, classifier tuning and model comparisons; a notebook, WEKA output PDF and 10–15-minute video. **The repository does not assert those deliverables have been completed.**

## Files

- `scraper.py` — adapted scraper with selectors from the actual pasted student code; supports local HTML for debugging and live Selenium browser execution.
- `prepare_data.py` — clean extracted price/bid fields and construct a **demonstration** classification target `high_price` (above/below sample median). This target is an *engineering choice made during recovery*, not part of the original assignment.
- `export_arff.py` — valid ARFF schema with nominal class for WEKA.
- `compare_models.py` — 3 classifier comparison (`LogisticRegression`, `RandomForest`, `SVC`) under baseline, correlation filtering and PCA. Stratified CV and preprocessing inside folds avoid obvious train/test leakage.
- `test_pipeline.py` — synthetic test data only; does not validate target website or original WEKA scores.
- `WEKA_GUIDE.md` — reproducible WEKA steps and comparison limitations.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
# Optional offline selector check: python scraper.py --html saved_page.html
# Only if allowed by the target site's terms and technical controls:
python scraper.py --output emirates_auction_data.csv
python prepare_data.py emirates_auction_data.csv --output model_input.csv
python export_arff.py model_input.csv --output model_input.arff
python compare_models.py model_input.csv --output model_comparison.csv
```

No CAPTCHA bypass, authenticated extraction, bulk requests, or real-data claims. Live selectors may no longer match the site's layout. The parser writes a **semicolon-delimited** CSV to match the original posted script; the model input CSV is comma-delimited. Do not publish unauthorized third-party records.

## Modeling caveats

- The original assignment did not provide a classification label. `high_price` is derived from price; **price and sequence are excluded from model inputs** to reduce leakage.
- A tiny collection, duplicated vehicle records, or highly imbalanced data cannot produce meaningful model evaluation.
- `remaining_minutes` is parsed only when the countdown format is recognized; unknown times are imputed inside CV.
- Report actual cross-validation metrics only after running with a permitted real dataset; synthetic test values are not portfolio findings.
