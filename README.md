# Kaggle Competition Solutions

Solution notebooks for two Kaggle competitions, built with stratified
cross-validation and gradient-boosting ensembles. Both notebooks are written
to run on Kaggle — attach the competition data and run all cells.

## Competitions & results

| Competition | Metric | v1 score | v2 score |
|---|---|---|---|
| [Predicting Credit Card Fraud — Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11) | ROC AUC | 0.94650 | pending (v2 running) |
| [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | Accuracy | 0.75837 | 0.77272 |

## 1. Predicting Credit Card Fraud (S6E11)

Predict the probability that a transaction is fraud (`is_fraud`).
~700,000 train rows, ~300,000 test rows, fraud rate ~1.7%.

**Approach (v2 notebook):**
- Feature engineering: log amounts/balances, amount-per-balance,
  cyclical hour/day-of-week encoding, frequency encoding of categoricals,
  and interactions (amount x risk, amount x velocity, foreign x VPN, …).
- Stratified 5-fold cross-validation.
- LightGBM + XGBoost (2 seeds each) + CatBoost, rank-averaged blend.
- Final submission: average of the three models' test probabilities.

## 2. Titanic — Machine Learning from Disaster

Predict passenger survival (`Survived`). 891 train rows, 418 test rows.

**Approach (v2 notebook):**
- Feature engineering: Title from Name, FamilySize, IsAlone, Deck from
  Cabin, FarePerPerson, TicketPrefix, AgeGroup, FamilySizeBin, Sex x Pclass.
- Missing values: Age by Sex x Pclass median, Fare by Pclass median,
  Embarked by mode.
- Stratified 5-fold cross-validation.
- Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost —
  blended by plain average vs OOF-accuracy-weighted average, whichever
  wins on out-of-fold accuracy.

## Files

- `s6e11-fraud-solution-v2.ipynb` — fraud solution (v2)
- `titanic-solution-v2.ipynb` — Titanic solution (v2)

Competition data is **not** included here — it comes from the Kaggle
competition pages linked above.

## Author

Bilal Ahmad — Healthcare Data Analyst | [LinkedIn](https://www.linkedin.com/in/bilal-ahmad-7ab444250) | [Kaggle](https://www.kaggle.com/bilalmustaqeem)
