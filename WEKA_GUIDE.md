# WEKA steps — course requirements

1. With actual collected and cleaned data, execute `export_arff.py` and ensure that `model_input.arff` begins with `@relation`. If you see an `@relation expected` error, you probably selected a CSV as ARFF or have a malformed header.
2. Open WEKA → Explorer → Preprocess → Open file → `model_input.arff`. Verify the final attribute `high_price` is **Nominal**, then set it as the class. `high_price` is a recovered-project demonstration choice, **not an original assignment target**.
3. Record baseline data counts and class balance; apply `ReplaceMissingValues`. For other filter experiments, test `Normalize`/`Standardize`, and supervised attribute selection. To prevent evaluation leakage, use a **FilteredClassifier** to fit filters within cross-validation, rather than preprocessing the entire dataset before CV.
4. Classify → choose J48, NaiveBayes, RandomForest, SMO; for each, choose cross-validation (same 10 folds or a consistent documented split) and save accuracy, weighted/macro precision, recall, F1, confusion matrix and settings.
5. Compare those results with `model_comparison.csv` **with clear notation**: Python uses LogisticRegression, RandomForest and SVM, so only RandomForest and SVM are like-for-like by model family. WEKA and scikit-learn defaults, preprocessing and folds differ, so raw numerical comparisons are not direct parity tests.
6. Save real screenshots to `evidence/` only after running WEKA. Do not submit fabricated output or synthetic test scores as field findings.

**Original project rubric:** minimum 500 scraped instances; Python preprocessing, ≥3 models, correlation-based feature selection, PCA, WEKA preprocessing/filtering and tuning, comparative analysis, notebook, WEKA PDF evidence and recorded walkthrough. The current reconstruction is a technically testable foundation and does not claim all rubric items are complete.
