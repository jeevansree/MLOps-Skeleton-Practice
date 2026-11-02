# Citi Bike Jersey City: Member Lookalike Scoring

> Citi Bike should invest in member-lookalike scoring as an MLOps project because retraining, quality gates and fallbacks handle seasonal drift, instead of one-off notebooks.

## 1. Business case

### Context
Client: Citi Bike (operated by Lyft), urban mobility and bike-share, Jersey City and Hoboken marketing team.

| Area | Rating | Why |
| --- | --- | --- |
| Data | HIGH | Every trip is logged automatically by docks and bikes and published monthly. In August and September 2026, 99.3% of 217,855 trips survived cleaning, with no duplicate trip IDs and station IDs matching names one-to-one. |
| Tools | MEDIUM | Citi Bike publishes a live station-status feed and monthly trip files. Nothing in the data shows an automated ML pipeline, so our Python pipeline (pandas and scikit-learn, run with one command) would be new to them. |
| Processes | MEDIUM | Membership marketing is established, but nothing shows offers are targeted using riding behaviour. |
| People | MEDIUM | Lyft has data teams that could maintain a Python pipeline. Local marketing staff would need results as ranked lists or dashboards rather than notebooks. |

Owner after delivery: Citi Bike's (Lyft's) data and analytics team runs and maintains the pipeline; the Jersey City marketing team uses its outputs.

### Problem and baseline
Citi Bike's Jersey City marketing team struggles with converting regular casual riders into members, and spends promotion budget on riders unlikely to convert, because nothing currently tells them which casual riders already ride like members.

Today they promote memberships broadly through the app, website and stations, without targeting by how or where people ride (assumption, not verified).

Baseline: casual-to-member conversion rate = To be measured per month. Source: Lyft's account records (share of casual riders who buy a membership within 30 days of a ride). Context from our data: casual riders take 27.1% of trips, about 29,500 per month (59,035 over August and September 2026, Citi Bike System Data).

### Goal and success metrics
Move the casual-to-member conversion rate from the measured baseline to 20% above baseline among targeted riders within 6 months, measured against a random holdout group that gets no targeted offer.

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Casual-to-member conversion rate (30 days) | To be measured | +20% vs holdout group within 6 months (proposed) |
| ML metric (quality gate) | `roc_auc` on unseen data | 0.5 (random ranking) | >= 0.68 |
| Limit | Response time per prediction | n/a | < 1 s (monthly batch) |
| Limit | Cost per prediction | n/a | < EUR 0.001 (proposed) |

Why `roc_auc`: offers go to the casual riders the model ranks as most member-like, so better ranking means offers reach riders more likely to convert. Gate reasoning: 0.68 is the level of the simplest model tested (logistic regression, 0.682); the selected model must not fall below it.

### Solution
Citi Bike's Jersey City marketing manager receives a monthly list of casual riders whose trips look most like members' trips, ranked from most to least member-like, to decide who gets a membership offer and when.

The model predicts whether a trip was made by a member or a casual rider from when and how the trip was ridden (hour, day of week, weekend or weekday, trip length, round trip or not, classic or e-bike) and where (how central the start and end stations are). It runs as a monthly batch.

If the model is unavailable or unsure: casual riders scoring below 0.8 are not targeted, and if the monthly run fails or fails the quality gate (`roc_auc` below 0.68), marketing goes back to today's broad membership promotion until the next successful run.

### Data
- **Source:** Citi Bike System Data, Jersey City trip files (public, published by Citi Bike). Owner: Lyft, which operates Citi Bike. Size: 217,855 trips (216,422 after cleaning), August to September 2026, one row per trip, 13 columns.
- **Label:** `member_casual`, recorded by the system on every trip from how it was paid for (membership or single ride/day pass), so no manual labelling is needed. Refresh: monthly, as a new trip file is published each month.
- **Limitation:** the public files have no rider ID, so the model scores trips, not riders. Turning scores into a list of riders depends on Lyft matching trips to accounts in its internal records; without that step, the output stays at trip level.
- **Personal or sensitive data:** YES, indirectly. The public files hold no names or rider IDs, but exact start times, stations and e-bike GPS coordinates can reveal a person's routine: the same trip between the same two stations every weekday at 8:05 points to a home and a workplace. Protected by: (1) the model uses only coarse features (hour, day, trip length, bike type, station centrality), never exact timestamps or coordinates; (2) linking trips to riders happens only inside Lyft's own systems, using pseudonymised account IDs; (3) the list sent to marketing contains only an account ID and a score, no trip histories; (4) access is limited to the marketing team, and lists are deleted after each monthly campaign.

### Value, cost and risks
**Benefit over a non-AI approach:** compared with the simple rule "target every casual rider who takes a weekday rush-hour trip", the model ranks riders far better (`roc_auc` 0.73 vs 0.55 for the rule on the same test set) and gives a shorter list (1,697 casual test trips at p >= 0.8, against about 3,200 flagged by the rule). Measured by: conversion rate per offer sent, model list vs rule list, in the holdout test.

**Rough cost (estimates):** 1 data scientist (about 3 weeks: time-based validation, setting the cutoff), 1 data engineer (about 3 weeks: monthly pipeline, scheduling, data checks), 1 marketing analyst (about 2 weeks: campaign setup and holdout test), plus Lyft data and privacy staff to link trips to accounts. Infrastructure: a scheduled Python job on a small cloud machine (low hundreds of euros a month). About 8 weeks to the first campaign, then 3 months of holdout testing.

| Risk | Mitigation |
| --- | --- |
| Data quality: missing, new or renamed stations break the station-centrality lookup | Automated checks before each monthly run (row count within 20% of the previous month, under 1% of trips missing a station ID, one name per station ID, new station IDs flagged); the run stops if a check fails |
| Model errors: offers go to riders who were never going to convert, wasting budget | Offers only above the 0.8 cutoff, list capped at a set budget, marketing reviews it before sending, holdout group shows wasted offers in the measured lift |
| Drift: a model trained on summer trips is applied to winter riding | Retrain monthly on the latest months, check `roc_auc` on the next unseen month, fall back to broad promotion if it drops below 0.68 |
| Skills: the pipeline was built by a student team | Tested modules, a runbook, and a walkthrough handover session with Lyft's data team |

**Scales to:** other Lyft-operated bike-share systems publishing trip data in the same format (Citi Bike New York, Divvy in Chicago, Bay Wheels in San Francisco, Capital Bikeshare in Washington DC), and other targeted offers such as e-bike or day-pass promotions.

### Open gaps
- Baseline conversion rate: to be measured from Lyft's account records
- Current marketing practice ("promote broadly") is an assumption, not verified
- Unseen-data test should use the next month's trips (time-based split); the notebook currently uses a random 20% split
- Trip-to-rider linking depends on Lyft's internal account data
- The +20% lift and cost limits are proposed targets, not measured

# MLOps Project Skeleton (reference only)

This repo is a **map, not a codebase**. Every file is a placeholder: no code, only a short note on
what the file does, where it sits in the pipeline and which files it talks to.

Create the same folders and files in **your own** project repo, then write the code for **your** use case.

## Stages = branches

| Branch | Stage | Sessions | What it adds |
|---|---|---|---|
| `main` (you are here) | 1. Kickoff | 1 to 3 | Folder structure, business case, one placeholder per pipeline step |
| `stage-2-modular` | 2. Modular code | 4 to 9 | Tests, style and test settings, first CI (Continuous Integration) workflow |
| `stage-3-cicd` | 3. CI/CD | 10 to 14 | Config loader, logging, experiment tracking, API, Docker, deployment |

Tip: on GitHub, open **Compare** (`main...stage-2-modular`) to see exactly what each stage adds.

## The pipeline at a glance

```text
                           main.py  (orchestrator, one command)
                              |
 data/raw/ -> load_data -> clean_data -> validate -> split -> features -> train -> evaluate
                                                (train/val/test) (recipe)   (fit)   (val split)
                                                                              |
                     reports/predictions.csv <- infer <- models/model.joblib (save)
```

## Stage 1 structure

```text
.
├── README.md                 # business case, success metrics, how to run
├── config.yaml               # non-secret settings (paths, target, split sizes)
├── environment.yml           # dependencies (conda)
├── .env.example              # names of secrets, never real values
├── .gitignore                # keeps data, secrets and generated files out of Git
├── LICENSE
├── .github/
│   └── pull_request_template.md
├── notebooks/                # the "lab": exploration only
├── src/                      # the "factory": one job per module
│   ├── load_data.py          # 1. read raw data
│   ├── clean_data.py         # 2. stateless cleaning
│   ├── validate.py           # 3. data quality gate
│   ├── features.py           # 4. unfitted feature recipe
│   ├── train.py              # 5. fit the pipeline
│   ├── evaluate.py           # 6. metrics on unseen data
│   ├── infer.py              # 7. predictions on new data
│   ├── utils.py              # shared read and write helpers
│   └── main.py               # 0. orchestrator: python -m src.main
├── data/  raw/ processed/ inference/
├── models/                   # saved model (generated)
├── reports/                  # predictions, plots (generated)
└── tests/                    # automated tests (stage 2)
```

## How to use this skeleton

1. Open a file and read its note
2. Create the equivalent file in your repo
3. Write your own code: the note describes the job, not the solution
