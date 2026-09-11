# Loan Approval Checker

## Live demo
https://project-loan-s3ef.onrender.com/

(Hosted on Render's free tier — the first request after a period of inactivity
can take 30-60 seconds to wake up.)

## Setup
```
pip install -r requirements.txt
python train_model.py   # trains the pipeline, writes model/loan_svc_model.pkl + model/model_metadata.json
python app.py            # starts the server at http://127.0.0.1:5000
```

## Pages
- `/`       - Step 2 dashboard: model config, metrics, confusion matrix, decision-boundary plot, feature list, sample rows
- `/apply`  - Step 3 form: enter applicant details, get an Approved / Not Approved decision

## API
- GET  /api/model/info
- GET  /api/model/features
- GET  /api/model/samples
- GET  /api/model/metrics
- GET  /api/model/decision_boundary
- POST /api/predict   body: {Age, Person Income, Loan Amount, Loan interest Rate, Credit Score, Home Onwership, Previous Loan}

## Loan Status label
Confirmed against the source dataset's documented column description (Kaggle
"Loan Approval Classification Data"): `loan_status` — 1 = approved, 0 = rejected.
`app.py` uses this mapping (`approved = prediction == 1`).

Note: this dataset's actual approval pattern is somewhat counterintuitive —
e.g. income correlates negatively with approval, and larger loan amounts
correlate with a higher approval rate. This is a property of the (synthetic)
source data, not a bug in the pipeline.
