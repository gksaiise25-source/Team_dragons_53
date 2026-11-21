import json
from main import load_artifacts, AnalyzeRequest, analyze_cashflow, get_benchmark, get_all_jobs, health_check

def run_tests():
    load_artifacts()

    print("=== TEST 1: HEALTH CHECK ===")
    health = health_check()
    print("Health:", health)
    assert health["model_loaded"] is True
    assert health["dataset_rows"] == 30

    print("\n=== TEST 2: DEMO 1 (Regular Earning Day: ₹1,800 - Safe Surplus) ===")
    req1 = AnalyzeRequest(
        current_date="2026-11-21",
        todays_earnings=1800.0,
        monthly_rent=8000.0,
        monthly_groceries=5500.0,
        monthly_emis=3000.0,
        already_reserved=21000.0,
        approaching_bills=2200.0,
        todays_basic_expenses=350.0,
        family_members=3,
        worker_skills="electrical wiring, appliance repair, plumbing",
        preferred_shift="Evening",
        max_distance_km=5.0
    )
    res1 = analyze_cashflow(req1)
    d1 = json.loads(res1.body.decode("utf-8"))
    print("Status Badge:", d1["protect"]["status_badge"])
    print("Amount to save today: ₹", d1["protect"]["amount_to_save_today"])
    print("Savings %:", d1["protect"]["savings_percentage"], "%")
    print("Safe to spend: ₹", d1["protect"]["safe_to_spend"])
    print("Shortage: ₹", d1["protect"]["shortage"])
    print("Recover triggered:", d1["recover"]["triggered"])

    assert d1["protect"]["amount_to_save_today"] > 0
    assert d1["protect"]["safe_to_spend"] > 0
    assert d1["protect"]["shortage"] == 0
    assert d1["protect"]["status_badge"] == "Fully Protected"

    print("\n=== TEST 3: DEMO 2 (Zero Earning Day: ₹0 Alert - Deficit & Gig Recovery) ===")
    req2 = AnalyzeRequest(
        current_date="2026-11-21",
        todays_earnings=0.0,
        monthly_rent=8000.0,
        monthly_groceries=5500.0,
        monthly_emis=3000.0,
        already_reserved=18000.0,
        approaching_bills=2200.0,
        todays_basic_expenses=350.0,
        family_members=3,
        worker_skills="electrical wiring, appliance repair, plumbing",
        preferred_shift="Evening",
        max_distance_km=5.0
    )
    res2 = analyze_cashflow(req2)
    d2 = json.loads(res2.body.decode("utf-8"))
    print("Status Badge:", d2["protect"]["status_badge"])
    print("Amount to save today: ₹", d2["protect"]["amount_to_save_today"])
    print("Safe to spend: ₹", d2["protect"]["safe_to_spend"])
    print("Shortage: ₹", d2["protect"]["shortage"])
    print("Recover triggered:", d2["recover"]["triggered"])
    assert d2["recover"]["triggered"] is True
    assert len(d2["recover"]["top_jobs"]) == 5

    print("\nTop 5 Recommended Gigs for ₹0 Earning Day:")
    for idx, job in enumerate(d2["recover"]["top_jobs"]):
        print(f"  #{idx+1}: [{job['job_id']}] {job['title']} | Score: {job['ml_match_score']:.1f}% | Dist: {job['distance_km']}km | Payout: ₹{job['payout_inr']}")

    print("\n=== TEST 4: BENCHMARK ENDPOINT ===")
    bm = get_benchmark()
    bm_data = json.loads(bm.body.decode("utf-8"))
    print("Benchmark models:", list(bm_data["benchmark_models"].keys()))

    print("\n>>> ALL TESTS COMPLETED & VERIFIED 100%! <<<")

if __name__ == "__main__":
    run_tests()
