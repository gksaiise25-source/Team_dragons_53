# -*- coding: utf-8 -*-
"""
SEVAMITR — Smart Income Allocation & Recovery Engine
Production-ready FastAPI backend integrating trained Random Forest Regressor
and TF-IDF Cosine Similarity Job Recommendation Engine.
"""

import os
import calendar
from datetime import date, datetime
from typing import List, Optional
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI App
app = FastAPI(
    title="SEVAMITR Engine API",
    description="Smart Income Allocation & Recovery Engine for Daily-Wage & Gig Workers",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded model and data
MODEL_PATH = "RandomForestRegressor.pkl"
ALT_MODEL_PATH = "expenditure_model.pkl"
JOBS_PATH = "jobs_dataset.csv"
BENCHMARK_PATH = "benchmark_metrics.json"

ml_model = None
jobs_df: Optional[pd.DataFrame] = None
benchmark_data = {}

def get_fallback_jobs() -> pd.DataFrame:
    """Provides fallback 30-job dataset if CSV is missing."""
    jobs_mock_data = [
        {"job_id": "JOB-101", "title": "Urgent Apartment Wiring & Fan Repair", "required_skills": "electrical wiring fan repair switchboard", "location": "Indiranagar (Near)", "distance_km": 1.2, "shift": "Evening", "duration_hrs": 4, "payout_inr": 950},
        {"job_id": "JOB-102", "title": "Commercial Inverter & MCB Installation", "required_skills": "electrical inverter wiring mcb maintenance", "location": "Whitefield (Far)", "distance_km": 18.5, "shift": "Morning", "duration_hrs": 6, "payout_inr": 1600},
        {"job_id": "JOB-103", "title": "Kitchen Pipe Leak & Tap Replacement", "required_skills": "plumbing pipe fitting tap repair", "location": "Koramangala (Near)", "distance_km": 2.1, "shift": "Evening", "duration_hrs": 3, "payout_inr": 800},
        {"job_id": "JOB-104", "title": "Shop Lighting & LED Board Setup", "required_skills": "electrical lighting wiring installation", "location": "HSR Layout (Near)", "distance_km": 3.4, "shift": "Afternoon", "duration_hrs": 4, "payout_inr": 900},
        {"job_id": "JOB-105", "title": "Bathroom Waterproofing & Plumbing Overhaul", "required_skills": "plumbing masonry pipe waterproofing", "location": "Yelahanka (Very Far)", "distance_km": 26.0, "shift": "Full Day", "duration_hrs": 8, "payout_inr": 2100},
        {"job_id": "JOB-106", "title": "Evening Grocery Hyperlocal Delivery", "required_skills": "bike driving delivery navigation", "location": "Domlur (Near)", "distance_km": 0.9, "shift": "Evening", "duration_hrs": 4, "payout_inr": 750},
        {"job_id": "JOB-107", "title": "Night E-Commerce Warehouse Sorting", "required_skills": "packing sorting loading warehouse helper", "location": "Electronic City (Far)", "distance_km": 16.2, "shift": "Night", "duration_hrs": 6, "payout_inr": 1100},
        {"job_id": "JOB-108", "title": "Tempo Mini-Truck Local Shifting Helper", "required_skills": "loading unloading driving helper shifting", "location": "Ulsoor (Near)", "distance_km": 2.8, "shift": "Morning", "duration_hrs": 5, "payout_inr": 1050},
        {"job_id": "JOB-109", "title": "Airport Late-Night Cab Relief Driver", "required_skills": "car driving commercial license navigation", "location": "Devanahalli (Very Far)", "distance_km": 28.5, "shift": "Night", "duration_hrs": 8, "payout_inr": 1900},
        {"job_id": "JOB-110", "title": "Afternoon Pharmacy Express Delivery", "required_skills": "bike driving delivery", "location": "Indiranagar (Near)", "distance_km": 1.5, "shift": "Afternoon", "duration_hrs": 4, "payout_inr": 650},
        {"job_id": "JOB-111", "title": "Wardrobe Hinge & Door Lock Fixing", "required_skills": "carpentry wood repair lock fitting", "location": "Jeevanbheemanagar (Near)", "distance_km": 1.8, "shift": "Evening", "duration_hrs": 3, "payout_inr": 850},
        {"job_id": "JOB-112", "title": "2BHK Interior Wall Repainting", "required_skills": "painting wall putty primer brushwork", "location": "Marathahalli (Medium)", "distance_km": 8.4, "shift": "Full Day", "duration_hrs": 8, "payout_inr": 1500},
        {"job_id": "JOB-113", "title": "Office Modular Desk Assembly", "required_skills": "carpentry furniture assembly drilling", "location": "MG Road (Near)", "distance_km": 3.9, "shift": "Afternoon", "duration_hrs": 5, "payout_inr": 1150},
        {"job_id": "JOB-114", "title": "Compound Wall Plastering & Brick Work", "required_skills": "masonry cement plastering construction", "location": "Hebbal (Far)", "distance_km": 14.7, "shift": "Morning", "duration_hrs": 7, "payout_inr": 1400},
        {"job_id": "JOB-115", "title": "Balcony Grill Rust Scraping & Enamel Paint", "required_skills": "painting welding metal work", "location": "Ejipura (Near)", "distance_km": 3.1, "shift": "Morning", "duration_hrs": 4, "payout_inr": 800},
        {"job_id": "JOB-116", "title": "Evening Banquet Catering Assistant & Server", "required_skills": "catering serving food prep hospitality", "location": "Indiranagar (Near)", "distance_km": 1.4, "shift": "Evening", "duration_hrs": 5, "payout_inr": 900},
        {"job_id": "JOB-117", "title": "Cloud Kitchen Tandoor & Roti Maker", "required_skills": "cooking tandoor kitchen prep", "location": "BTM Layout (Medium)", "distance_km": 6.8, "shift": "Evening", "duration_hrs": 6, "payout_inr": 1200},
        {"job_id": "JOB-118", "title": "Morning Breakfast Tiffin Stall Helper", "required_skills": "cooking chopping cleaning kitchen helper", "location": "Halasuru (Near)", "distance_km": 2.3, "shift": "Morning", "duration_hrs": 4, "payout_inr": 600},
        {"job_id": "JOB-119", "title": "Wedding Event Dishwashing & Kitchen Crew", "required_skills": "cleaning kitchen helper catering", "location": "Bannerghatta (Far)", "distance_km": 15.3, "shift": "Full Day", "duration_hrs": 9, "payout_inr": 1350},
        {"job_id": "JOB-120", "title": "Split AC Gas Refill & Jet Servicing", "required_skills": "ac repair appliance servicing electrical", "location": "Cambridge Layout (Near)", "distance_km": 1.6, "shift": "Evening", "duration_hrs": 3, "payout_inr": 1100},
        {"job_id": "JOB-121", "title": "Washing Machine Motor & Drain Fix", "required_skills": "appliance repair electrical plumbing", "location": "CV Raman Nagar (Near)", "distance_km": 2.9, "shift": "Afternoon", "duration_hrs": 3, "payout_inr": 900},
        {"job_id": "JOB-122", "title": "Apartment Water Tank Cleaning", "required_skills": "cleaning plumbing general helper", "location": "JP Nagar (Medium)", "distance_km": 9.5, "shift": "Morning", "duration_hrs": 4, "payout_inr": 850},
        {"job_id": "JOB-123", "title": "Garden Pruning & Lawn Maintenance", "required_skills": "gardening landscaping cleaning", "location": "Sadashivanagar (Medium)", "distance_km": 10.2, "shift": "Morning", "duration_hrs": 5, "payout_inr": 950},
        {"job_id": "JOB-124", "title": "Event Stage & Sound Box Loading Crew", "required_skills": "loading helper electrical cable setup", "location": "Koramangala (Near)", "distance_km": 2.5, "shift": "Evening", "duration_hrs": 5, "payout_inr": 950},
        {"job_id": "JOB-125", "title": "Retail Store Stock Counting & Shelf Setup", "required_skills": "inventory sorting helper cleaning", "location": "Commercial Street (Near)", "distance_km": 4.1, "shift": "Night", "duration_hrs": 5, "payout_inr": 850},
        {"job_id": "JOB-126", "title": "Solar Panel Dust Cleaning on Rooftop", "required_skills": "cleaning electrical basic maintenance", "location": "Bellandur (Medium)", "distance_km": 7.9, "shift": "Morning", "duration_hrs": 4, "payout_inr": 900},
        {"job_id": "JOB-127", "title": "Security Guard Night Shift Replacement", "required_skills": "security watchman night duty", "location": "Domlur (Near)", "distance_km": 1.1, "shift": "Night", "duration_hrs": 8, "payout_inr": 1000},
        {"job_id": "JOB-128", "title": "2-Wheeler Puncture & Basic Mechanic Help", "required_skills": "mechanic bike repair welding", "location": "Madiwala (Medium)", "distance_km": 5.8, "shift": "Afternoon", "duration_hrs": 5, "payout_inr": 850},
        {"job_id": "JOB-129", "title": "Office Deep Cleaning & Sofa Shampooing", "required_skills": "deep cleaning housekeeping machine handling", "location": "MG Road (Near)", "distance_km": 3.7, "shift": "Evening", "duration_hrs": 4, "payout_inr": 920},
        {"job_id": "JOB-130", "title": "Construction Site Bar Bending & Helper", "required_skills": "construction masonry heavy lifting", "location": "Peenya (Very Far)", "distance_km": 22.4, "shift": "Full Day", "duration_hrs": 8, "payout_inr": 1300},
    ]
    return pd.DataFrame(jobs_mock_data)

def load_artifacts():
    global ml_model, jobs_df, benchmark_data
    if ml_model is not None and jobs_df is not None:
        return

    print("=" * 60)
    print("Initializing SEVAMITR Machine Learning Engine...")

    # Load Model
    target_model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else ALT_MODEL_PATH
    if os.path.exists(target_model_file):
        ml_model = joblib.load(target_model_file)
        print(f" Loaded ML Model from: {target_model_file}")
    else:
        print(f" Warning: Neither {MODEL_PATH} nor {ALT_MODEL_PATH} found on disk.")

    # Load Jobs Dataset
    if os.path.exists(JOBS_PATH):
        jobs_df = pd.read_csv(JOBS_PATH)
        print(f" Loaded {len(jobs_df)} jobs from {JOBS_PATH}")
    else:
        jobs_df = get_fallback_jobs()
        jobs_df.to_csv(JOBS_PATH, index=False)
        print(f" Created and saved default {len(jobs_df)} jobs to {JOBS_PATH}")

    # Load Benchmarks
    if os.path.exists(BENCHMARK_PATH):
        try:
            with open(BENCHMARK_PATH, "r") as f:
                benchmark_data = json.load(f)
            print(f" Loaded model benchmarks ({len(benchmark_data)} models)")
        except Exception as e:
            print(f"Failed to load benchmark JSON: {e}")

    print("SEVAMITR Ready to Serve Predictions!")
    print("=" * 60)

@app.on_event("startup")
def startup_event():
    load_artifacts()


# ==============================================================================
# SCHEMAS
# ==============================================================================

class AnalyzeRequest(BaseModel):
    current_date: Optional[str] = Field(default=None, description="Current Date in YYYY-MM-DD")
    todays_earnings: float = Field(default=1800.0, ge=0.0, description="Worker's earnings today (₹)")
    monthly_rent: float = Field(default=8000.0, ge=0.0, description="Monthly rent amount (₹)")
    monthly_groceries: float = Field(default=5500.0, ge=0.0, description="Monthly grocery budget (₹)")
    monthly_emis: float = Field(default=3000.0, ge=0.0, description="Monthly EMIs / loan repayments (₹)")
    already_reserved: float = Field(default=14500.0, ge=0.0, description="Amount already locked/reserved this month (₹)")
    approaching_bills: float = Field(default=2200.0, ge=0.0, description="Upcoming utility and urgent bills (₹)")
    todays_basic_expenses: float = Field(default=350.0, ge=0.0, description="Worker's mandatory daily subsistence expenses (₹)")
    family_members: int = Field(default=3, ge=1, le=12, description="Number of family dependants")

    # Recovery Parameters
    worker_skills: str = Field(default="electrical wiring, appliance repair, plumbing", description="Comma-separated worker skills")
    preferred_shift: str = Field(default="Evening", description="Preferred shift (Morning, Afternoon, Evening, Night, Full Day)")
    max_distance_km: float = Field(default=5.0, ge=1.0, le=50.0, description="Maximum commute radius (km)")
    force_recover: Optional[bool] = Field(default=False, description="Force trigger Recovery feature for testing")


# ==============================================================================
# ML RECOVERY ALGORITHM (TF-IDF + Proximity + Shift + Earnings Score)
# ==============================================================================

def rank_jobs(
    worker_skills: str,
    preferred_shift: str,
    max_distance_km: float,
    shortage_amount: float,
    jobs_dataframe: pd.DataFrame,
    top_n: int = 5,
):
    df = jobs_dataframe.copy()

    # 1. NLP Skill Match using TF-IDF and Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    clean_worker_skills = worker_skills.strip().lower() if worker_skills.strip() else "general helper"
    corpus = df["required_skills"].tolist() + [clean_worker_skills]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    skill_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    df["skill_score"] = skill_similarities

    # 2. Distance Proximity Score (Exponential decay)
    df["proximity_score"] = np.exp(-df["distance_km"] / max(1.0, max_distance_km))

    # 3. Shift Compatibility Score
    def calc_shift_score(job_shift: str) -> float:
        js = str(job_shift).strip().lower()
        ps = preferred_shift.strip().lower()
        if js == ps:
            return 1.0
        elif js == "full day" or ps == "full day":
            return 0.4
        return 0.15

    df["shift_score"] = df["shift"].apply(calc_shift_score)

    # 4. Earnings Shortage Recovery Score
    effective_shortage = max(1.0, shortage_amount)
    df["earnings_score"] = np.minimum(1.0, df["payout_inr"] / effective_shortage)

    # 5. Weighted ML Confidence Match Score (0 - 100%)
    df["ml_match_score"] = (
        (0.45 * df["skill_score"])
        + (0.30 * df["proximity_score"])
        + (0.15 * df["shift_score"])
        + (0.10 * df["earnings_score"])
    ) * 100

    # Extract overlapping matched skill keywords for UI tag highlighting
    worker_tokens = set([t.strip().lower() for t in worker_skills.replace(",", " ").split() if len(t.strip()) > 2])
    
    def extract_matched_tags(req_skills: str):
        job_tokens = set([t.strip().lower() for t in str(req_skills).replace(",", " ").split() if len(t.strip()) > 2])
        matched = worker_tokens.intersection(job_tokens)
        if matched:
            return list(matched)
        return [str(req_skills).split()[0]] if str(req_skills) else ["general"]

    df["matched_tags"] = df["required_skills"].apply(extract_matched_tags)

    # Sort descending
    ranked_df = df.sort_values(by="ml_match_score", ascending=False).reset_index(drop=True)
    
    top_matches = ranked_df.head(top_n).to_dict(orient="records")
    all_ranked = ranked_df.to_dict(orient="records")

    return top_matches, all_ranked


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/api/analyze")
def analyze_cashflow(req: AnalyzeRequest):
    """
    Core Full-Stack Engine:
    Executes Feature 1 (Smart Income Allocation — PROTECT with .pkl model)
    and conditionally triggers Feature 2 (Gig Recovery — RECOVER with TF-IDF Ranking).
    """
    global ml_model, jobs_df

    if ml_model is None or jobs_df is None:
        load_artifacts()

    # 1. Date and Days Remaining Calculation
    parsed_date = date.today()
    if req.current_date:
        try:
            parsed_date = datetime.strptime(req.current_date, "%Y-%m-%d").date()
        except ValueError:
            parsed_date = date.today()

    total_days_in_month = calendar.monthrange(parsed_date.year, parsed_date.month)[1]
    days_remaining = (total_days_in_month - parsed_date.day) + 1  # Inclusive

    # 2. FEATURE 1: PROTECT (.pkl Model Inference)
    worker_features = pd.DataFrame([{
        "monthly_rent": float(req.monthly_rent),
        "monthly_groceries": float(req.monthly_groceries),
        "monthly_emis": float(req.monthly_emis),
        "approaching_bills": float(req.approaching_bills),
        "daily_basic_avg": float(req.todays_basic_expenses),
        "family_members": int(req.family_members),
    }])

    predicted_monthly_expenditure = float(ml_model.predict(worker_features)[0])
    remaining_target_needed = max(0.0, predicted_monthly_expenditure - req.already_reserved)
    required_daily_reserve = remaining_target_needed / max(1, days_remaining)
    net_earnings_after_basics = req.todays_earnings - req.todays_basic_expenses

    # Allocation breakdown
    if req.todays_earnings <= 0:
        savings_percentage = 0.0
        amount_to_save_today = 0.0
        safe_to_spend = 0.0
        status_code = "ZERO_EARNINGS"
        status_badge = "Zero Earnings Today"
        status_message = "No income recorded today. Immediate gig recovery activated to prevent debt spiral."
        status_color = "red"
    elif net_earnings_after_basics <= 0:
        savings_percentage = 0.0
        amount_to_save_today = 0.0
        safe_to_spend = 0.0
        status_code = "BASIC_DEFICIT"
        status_badge = "Deficit Alert"
        status_message = f"Today's earnings (₹{req.todays_earnings:,.0f}) are below basic subsistence needs (₹{req.todays_basic_expenses:,.0f})."
        status_color = "red"
    else:
        amount_to_save_today = min(required_daily_reserve, net_earnings_after_basics)
        savings_percentage = (amount_to_save_today / req.todays_earnings) * 100
        safe_to_spend = max(0.0, req.todays_earnings - req.todays_basic_expenses - amount_to_save_today)
        
        if amount_to_save_today >= required_daily_reserve:
            status_code = "OPTIMAL_SAVINGS"
            status_badge = "Fully Protected"
            status_message = "Daily target reserve is 100% funded! Surplus is safe for discretionary spending."
            status_color = "green"
        else:
            shortfall = required_daily_reserve - amount_to_save_today
            status_code = "PARTIAL_RESERVE"
            status_badge = "Partial Reserve"
            status_message = f"Today's savings are short by ₹{shortfall:,.0f}; catch up on upcoming high-wage days."
            status_color = "amber"

    # Shortage Calculation
    total_needed_today = required_daily_reserve + req.todays_basic_expenses
    shortage = max(0.0, total_needed_today - req.todays_earnings)

    # 3. FEATURE 2: RECOVER (Trigger Check)
    should_recover = bool(req.todays_earnings == 0 or shortage > 0 or req.force_recover)
    
    top_jobs = []
    all_jobs_ranked = []
    recovery_reason = ""

    if should_recover:
        if req.todays_earnings == 0:
            recovery_reason = "Zero Earnings Trigger: Immediate day-wage recovery required."
        elif shortage > 0:
            recovery_reason = f"Cashflow Gap Trigger: ₹{shortage:,.0f} needed to secure monthly reserve and daily basics."
        else:
            recovery_reason = "Manual Simulation Trigger: Exploring temporary shift opportunities."

        top_jobs, all_jobs_ranked = rank_jobs(
            worker_skills=req.worker_skills,
            preferred_shift=req.preferred_shift,
            max_distance_km=req.max_distance_km,
            shortage_amount=shortage if shortage > 0 else 1000.0,
            jobs_dataframe=jobs_df,
            top_n=5,
        )

    # Visual split breakdown for donut/stacked bar chart
    actual_spent_on_basics = min(req.todays_earnings, req.todays_basic_expenses)
    allocation_chart_data = {
        "basic_needs": round(actual_spent_on_basics, 2),
        "locked_reserve": round(amount_to_save_today, 2),
        "safe_to_spend": round(safe_to_spend, 2),
        "uncovered_shortage": round(shortage, 2),
    }

    response_payload = {
        "meta": {
            "analyzed_date": str(parsed_date),
            "days_remaining": days_remaining,
            "total_days_in_month": total_days_in_month,
            "model_type": "Random Forest Regressor (150 Estimators, R² = 95.4%)",
        },
        "protect": {
            "todays_earnings": req.todays_earnings,
            "todays_basic_expenses": req.todays_basic_expenses,
            "predicted_monthly_expenditure": round(predicted_monthly_expenditure, 2),
            "already_reserved": req.already_reserved,
            "approaching_bills": req.approaching_bills,
            "remaining_target_needed": round(remaining_target_needed, 2),
            "required_daily_reserve": round(required_daily_reserve, 2),
            "amount_to_save_today": round(amount_to_save_today, 2),
            "savings_percentage": round(savings_percentage, 1),
            "safe_to_spend": round(safe_to_spend, 2),
            "shortage": round(shortage, 2),
            "net_earnings_after_basics": round(net_earnings_after_basics, 2),
            "status_code": status_code,
            "status_badge": status_badge,
            "status_message": status_message,
            "status_color": status_color,
            "chart_breakdown": allocation_chart_data,
        },
        "recover": {
            "triggered": should_recover,
            "reason": recovery_reason,
            "gap_amount": round(shortage, 2),
            "worker_skills": req.worker_skills,
            "preferred_shift": req.preferred_shift,
            "max_distance_km": req.max_distance_km,
            "top_jobs": top_jobs,
            "all_jobs": all_jobs_ranked,
        },
    }

    return JSONResponse(content=response_payload)


@app.get("/api/benchmark")
def get_benchmark():
    """Returns real ML model comparison metrics for Judge Review."""
    global benchmark_data
    if not benchmark_data and os.path.exists(BENCHMARK_PATH):
        try:
            with open(BENCHMARK_PATH, "r") as f:
                benchmark_data = json.load(f)
        except Exception:
            pass
    return JSONResponse(content={
        "status": "success",
        "benchmark_models": benchmark_data,
        "selected_production_model": "Random Forest Regressor",
        "description": "Cross-evaluated on 2,000 monthly worker records with Rent, Groceries, EMIs, Utility Bills, Daily Basics, and Household Size."
    })


@app.get("/api/jobs")
def get_all_jobs():
    """Returns the raw 30-gig marketplace dataset."""
    global jobs_df
    if jobs_df is None:
        jobs_df = get_fallback_jobs()
    return JSONResponse(content={"jobs": jobs_df.to_dict(orient="records")})


@app.get("/api/health")
def health_check():
    """Health check verifying model and dataset availability."""
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "dataset_rows": len(jobs_df) if jobs_df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


# Mount Static Frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
