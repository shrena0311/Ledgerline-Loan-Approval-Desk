# Ledgerline — Loan Approval Desk

A Streamlit app for the Loan Eligibility Predictor notebook, styled as a bank
loan officer's ledger desk rather than a generic ML dashboard.

## What it does

- Retrains the notebook's tuned Logistic Regression model (plus Naive Bayes
  and Random Forest for comparison) on `loan_data.csv` the first time the app
  starts, then caches it.
- Lets you fill out an "applicant file" and get back an approval probability,
  a risk grade (Low / Medium / High), and a stamped decision slip.
- Includes an expandable "Behind the decision" panel with the logistic
  regression's feature weights and a model comparison table.
- Three "quick fill" buttons prefill a strong, borderline, and risky sample
  applicant so you can see the range of outcomes immediately.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `loan_data.csv` stays in the same folder as `app.py` — the app
reads it at startup to train the models.

## Files

- `app.py` — the Streamlit app
- `loan_data.csv` — the training data (same rows as `Loan_Data.xls`, just
  saved with a `.csv` extension since the original file was already
  comma-separated text)
- `requirements.txt` — Python dependencies
