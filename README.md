# Predicting hourly bike rentals in Washington, D.C.

MSCS 634 · Residency Day 2 · Project Deliverable 2

Student: Rohail Shaikh, Phaneendra Vanapamula, Abhilash Marthala Course: Advanced Big Data and Data Mining (MSCS-634-M50)

## Data and question

How accurately can calendar and weather information predict total systemwide rentals in a recorded hour? The included Day 1 export contains 17,379 hourly observations and 17 columns from 2011–2012. The target is `total_rentals`. Recorded counts range from 1 to 977. The model assumes hourly weather is known.

## Modeling design

- Compare a training-mean baseline, additive Ridge regression, Ridge with engineered calendar features, and histogram gradient boosting with the expanded features.
- Add cyclic hour coordinates, an hour-by-working-day category, and elapsed days. Separate hourly profiles let the model capture commuting peaks and midday weekend activity.
- Use four expanding-window validation folds from Q4 2011 through Q3 2012. Fit all preprocessing on each training window. Select by mean quarterly RMSE, then refit on 15,211 hours before October 2012 and evaluate on 2,168 hours in Q4 2012.

## Results

| Model | Mean CV RMSE | CV RMSE SD | Test R² | Test MSE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Boosting / engineered | 73.41 | 19.44 | 0.780 | 8,950.01 | 94.60 |
| Ridge / engineered | 104.61 | 19.31 | 0.773 | 9,214.86 | 95.99 |
| Ridge / original | 168.02 | 48.58 | 0.197 | 32,638.84 | 180.66 |
| Training mean | 198.31 | 64.40 | -0.028 | 41,790.95 | 204.43 |

**Boosting / engineered** had the lowest mean validation RMSE (73.41 rentals/hour). On the final quarter its RMSE was **94.60**, with R² of **0.780**. The lowest test RMSE belonged to **Boosting / engineered** (94.60).

Adding calendar features reduced mean validation RMSE for Ridge by **63.41 rentals/hour (37.7%)**. Because both Ridge fits used identical folds and penalties, the change measures the combined contribution of the feature set. It does not isolate a single feature.

In the busiest tenth of held-out hours, observed demand averaged **660.2** rentals/hour and predictions averaged **648.5**. A bike allocation plan based solely on that estimate could leave too few bikes available during peaks. These scores describe one city's recorded hours and one final quarter. Future weather forecast errors and changes in ridership could affect performance.

## Challenges and decisions

- **Target leakage.** Casual and registered rental counts sum exactly to the target. An explicit predictor list excludes both.
- **Irregular coverage.** There are 165 unrecorded hours in the two-year period. Splits use calendar boundaries, and unknown counts remain absent.
- **Changing demand.** Ridership grew across the two years. A chronological holdout tests how well each model transfers to a later period. The elapsed-days feature lets Ridge estimate a trend, while trees can learn thresholds only within the observed training range.
- **Rare peaks.** Retaining high counts preserves the hours that matter for capacity planning. The notebook checks peak errors directly with a held-out demand-band plot.

## Reproduce

Keep the notebook, CSV and requirements file in the same folder. With Python 3.12, run:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab Regression_Modeling.ipynb
```

Choose **Restart Kernel and Run All Cells**. The notebook includes the code, fold-by-fold results, held-out comparison and three diagnostic plots. Its saved outputs can also be viewed directly on GitHub.

## Data attribution

Hadi Fanaee-T, *Bike Sharing*, UCI Machine Learning Repository. The included CSV is the Day 1 adaptation of `hour.csv`, with descriptive column names, calendar years, and timestamps. The original record ID was removed. [Dataset and DOI](https://doi.org/10.24432/C5W894) · [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
