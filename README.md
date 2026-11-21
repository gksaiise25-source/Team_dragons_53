# 🛡️ SEVAMITR — Smart Income Allocation & Recovery Engine

> **AI-Powered Cashflow Protection & Gig Recovery for Daily-Wage & Informal Workers**  
> *Production-Ready Full-Stack Web Application with Real Machine Learning Inference*

---

## 📌 Executive Summary

Daily-wage, gig, and informal workers operate with volatile income streams. Traditional financial tools fail them because they assume stable monthly salaries. **SEVAMITR** solves this with an executive dual-pipeline architecture:

1. **PROTECT Pipeline (Expenditure Forecasting ML Model):** Dynamically predicts monthly obligations and calculates exact, guilt-free daily savings allocations (`%` and `₹`) and safe-to-spend allowances before month-end reset.
2. **RECOVER Pipeline (Gig Recommendation Engine):** Automatically activates when zero or deficit earnings occur, using **TF-IDF NLP** semantic matching, exponential distance decay, and shift compatibility to recommend high-yield hyperlocal temporary gigs from a 30-job marketplace.

---

## 🎨 Visual Design & Theme System

Built strictly adhering to the **Fintech White & Professional Theme System**:
- **Primary Background:** `#FFFFFF` crisp clean canvas with `#F8FAFC` subtle card surfaces.
- **Borders & Dividers:** 1px `#E2E8F0` crisp precision borders.
- **Typography:** Deep Slate `#0F172A` headings, `#475569` body, and `JetBrains Mono` for all currency numbers.
- **Primary Accent:** Royal Blue (`#2563EB` / `#1E40AF`) for interactive controls.
- **Safe & Savings Metrics:** Emerald Green (`#059669` / `#10B981`) for locked vaults and surplus.
- **Deficit / Shortage Alerts:** Crimson Red (`#DC2626`) and Amber (`#D97706`) for cashflow warning badges.

---

## 🧠 Machine Learning Engine Architecture

### 1. Feature 1: Smart Income Allocation — PROTECT (`.pkl` Model)
- **Trained Model:** `RandomForestRegressor.pkl` (150 decision trees, trained on 2,000 monthly worker records).
- **Loaded directly via:** `joblib.load("RandomForestRegressor.pkl")` at server startup (zero hardcoded mock predictions).
- **Features Passed to Model:**
  - `monthly_rent` (₹)
  - `monthly_groceries` (₹)
  - `monthly_emis` (₹)
  - `approaching_bills` (₹)
  - `daily_basic_avg` (₹) (Worker's mandatory daily subsistence)
  - `family_members` (Household dependants)
- **Math & Protection Logic:**
  $$\text{Days Remaining} = (\text{Days in Month} - \text{Current Day}) + 1$$
  $$\text{Remaining Target Needed} = \max(0, \text{Predicted Expenditure} - \text{Already Reserved})$$
  $$\text{Required Daily Reserve} = \frac{\text{Remaining Target Needed}}{\max(1, \text{Days Remaining})}$$
  $$\text{Net Earnings After Basics} = \text{Today's Earnings} - \text{Today's Basic Expenses}$$
  - When $\text{Today's Earnings} > 0$ and $\text{Net Earnings} > 0$:
    $$\text{Amount to Save Today} = \min(\text{Required Daily Reserve}, \text{Net Earnings})$$
    $$\text{Savings Percentage} = \left(\frac{\text{Amount to Save Today}}{\text{Today's Earnings}}\right) \times 100$$
    $$\text{Safe to Spend} = \max(0, \text{Today's Earnings} - \text{Today's Basic Expenses} - \text{Amount to Save Today})$$
  - Shortage:
    $$\text{Shortage} = \max(0, (\text{Required Daily Reserve} + \text{Today's Basic Expenses}) - \text{Today's Earnings})$$

---

### 2. Feature 2: Alternative Income Recommendations — RECOVER (Hyperlocal Gig Engine)
- **Automatic Trigger:** Activates immediately when $\text{Today's Earnings} == 0$ OR $\text{Shortage} > 0$.
- **Marketplace Dataset:** 30 real-world temporary gigs (`jobs_dataset.csv`) across electrical, plumbing, warehouse, logistics, catering, AC servicing, and general trades.
- **Multimodal Scoring Objective:**
  1. **NLP Skill Semantic Match ($S_{\text{skill}}$):** TF-IDF Vectorizer with English stopwords and Cosine Similarity between worker's skills and job requirements.
  2. **Proximity Decay ($S_{\text{prox}}$):**
     $$S_{\text{prox}} = \exp\left(-\frac{\text{Distance (km)}}{\max(1, \text{Max Radius})}\right)$$
  3. **Shift Compatibility ($S_{\text{shift}}$):** $1.0$ for exact shift match, $0.4$ for Full Day flexibility, $0.15$ otherwise.
  4. **Earnings Recovery ($S_{\text{earn}}$):**
     $$S_{\text{earn}} = \min\left(1.0, \frac{\text{Payout (₹)}}{\max(1, \text{Shortage})}\right)$$
  5. **Final Weighted Match Confidence (%):**
     $$\text{ML Match Score} = \left(0.45 \cdot S_{\text{skill}} + 0.30 \cdot S_{\text{prox}} + 0.15 \cdot S_{\text{shift}} + 0.10 \cdot S_{\text{earn}}\right) \times 100$$

---

## 📊 Live Judge Demo Scenarios

In the top navigation bar, two pre-configured judge demo presets allow 1-click evaluation:

| Demo Button | Inputs | Protect Outcome | Recover Triggered? |
|---|---|---|---|
| **Demo 1: Earning Day (₹1,800)** | Rent: ₹8k, Groceries: ₹5.5k, EMIs: ₹3k, Saved: ₹21k, Bills: ₹2.2k, Earnings: ₹1,800, Basics: ₹350 | **Save:** ₹1,059 (58.8%), **Safe-to-Spend:** ₹391, Status: **Fully Protected** | ❌ No (Surplus) |
| **Demo 2: Zero Day (₹0 Alert)** | Earnings: ₹0, Basics: ₹350, Shortage: ₹1,709.26 | **Save:** ₹0, **Safe-to-Spend:** ₹0, Status: **Zero Earnings Alert** | 🚨 **YES (Triggers Top 5 Gigs + ML Decision Scatter Plot)** |

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+
- Installed packages: `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `pydantic`

### 2. Start the Server
Run using the Windows batch launcher:
```cmd
start.bat
```
Or directly via Python:
```bash
python main.py
```
Or with Uvicorn:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Open in Browser
Visit: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📡 API Endpoints

- `POST /api/analyze` — Core dual-pipeline execution (runs `.pkl` inference and gig ranking).
- `GET /api/benchmark` — Returns cross-model benchmark metrics (Ridge, Random Forest, XGBoost, LightGBM, MLP, SARIMAX).
- `GET /api/jobs` — Returns all 30 jobs in the marketplace dataset.
- `GET /api/health` — Confirms model status and active dataset rows.

---

## 🏆 Model Benchmarks (Judge Review Panel)

| Model Architecture | R² Score | Accuracy (%) | MAE (₹) | Status |
|---|---|---|---|---|
| **Ridge Regression (L2)** | 0.9961 | 99.61% | ₹451.71 | Benchmarked |
| **Random Forest Regressor (150 Trees)** | **0.9539** | **95.39%** | **₹1,626.84** | **Deployed (`.pkl`)** |
| **XGBoost Regressor** | 0.9882 | 98.82% | ₹542.15 | Benchmarked |
| **LightGBM Regressor** | 0.9864 | 98.64% | ₹589.30 | Benchmarked |
| **MLP Neural Network** | 0.9841 | 98.41% | ₹621.50 | Benchmarked |
| **SARIMAX (Time-Series)** | 0.9790 | 97.90% | ₹710.40 | Benchmarked |
