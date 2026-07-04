# Bank Term Deposit Prediction

Predicting which bank customers are likely to subscribe to a term deposit, using the UCI Bank Marketing dataset — with the goal of helping the marketing team call the right people instead of everyone.

## Business Context

A bank runs outbound phone campaigns to sell term deposits (a fixed savings product). Calling every customer is expensive and mostly unproductive — only a small fraction ever say yes. This project analyzes past campaign data to answer a practical question for the business: **who should we actually be calling?**

The output isn't just a model — it's a way to rank customers by likelihood to subscribe, so calling lists can be prioritized instead of worked top-to-bottom at random.

**Dataset:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) — 45,211 customer records from a Portuguese bank's phone campaigns.

## What's in the Data

| Category | Columns |
|---|---|
| Customer profile | age, job, marital, education |
| Financial standing | default, balance, housing, loan |
| Campaign contact details | contact, day, month, duration, campaign |
| Campaign history | pdays, previous, poutcome |
| Target | y (subscribed: yes/no) |

No missing values or duplicates — the main data quality issue was outliers in `balance`, `duration`, and `campaign`, handled with IQR-based checks.

## Approach

1. **Cleaning & outlier checks** — IQR method on balance, duration, and campaign count.
2. **Encoding** — binary mapping for yes/no columns, one-hot encoding for categorical fields (job, marital, education, contact, poutcome, month).
3. **Feature engineering** — added `was_contacted_before`, `age_group` buckets, and a combined `contact_intensity` metric (current + past campaign contacts).
4. **Statistical testing** — chi-square tests on categorical variables, t-tests on numeric variables against the target, to confirm which factors are actually statistically associated with subscribing before feeding them into a model.
5. **Modeling** — trained and compared 5 classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes) with class-weight balancing, since only ~12% of customers subscribe — a heavily imbalanced target.
6. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, and ROC-AUC, with the *business* metric (catching actual subscribers) prioritized over raw accuracy.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Naive Bayes** | 0.85 | 0.38 | 0.44 | **0.41** |
| Logistic Regression | 0.77 | 0.28 | 0.64 | 0.39 |
| Random Forest | 0.89 | 0.62 | 0.20 | 0.30 |
| Decision Tree | 0.84 | 0.31 | 0.29 | 0.30 |
| KNN | 0.88 | 0.49 | 0.17 | 0.26 |

**Why Naive Bayes, not the highest-accuracy model:** Random Forest and KNN post higher accuracy, but that's mostly because they're good at predicting "no" — the majority class. For a marketing team, missing a real subscriber (false negative) is a bigger cost than accuracy on paper suggests. Naive Bayes gives the best balance of catching actual subscribers (recall) without flooding the list with wrong calls, which is why it wins on F1 score, the metric that matters most here.

## Key Insights

1. **Past campaign response is the single strongest predictor.** Customers who said yes in a previous campaign are far more likely to say yes again — this is by far the most useful signal in the data.
2. **Contact intensity and prior contact history both raise subscription likelihood** — customers who've been reached before, and reached more, convert at higher rates.
3. **Demographic and financial variables (age, balance, job, etc.) are all statistically significant, but weak individually** — they help fine-tune the ranking, but campaign history is doing the heavy lifting.

## Business Recommendations

1. **Prioritize customers with a prior successful campaign outcome** for outreach — they're the highest-probability segment by a wide margin.
2. **De-prioritize customers with no engagement signal** (never contacted, no response history) to cut wasted call volume.
3. **Sequence calling lists by model-ranked probability**, not randomly or alphabetically — this turns the model into an operational tool, not just an analysis.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scipy (chi-square, t-tests)
- scikit-learn (modeling, evaluation)


## How to Run

```bash
git clone https://github.com/DishaBisht75/bank-term-deposit-prediction.git
cd bank-term-deposit-prediction
pip install -r requirements.txt
jupyter notebook notebooks/bank_marketing_analysis.ipynb
```

## Limitations

- The target is heavily imbalanced (~88% no / ~12% yes), which caps how high precision and recall can realistically get without more advanced resampling techniques.
- `duration` (call length) is a strong predictor in the raw data but isn't known before a call happens, so it was excluded from modeling to keep the model realistic for actual campaign planning.
- Findings reflect one bank's campaign in one time period — behavior may not generalize to other markets or products.

## Author

Disha Bisht - www.linkedin.com/in/disha-bisht-00255935a
