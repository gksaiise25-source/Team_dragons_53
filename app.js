/**
 * SEVAMITR — Frontend Application Logic
 * Integrates with FastAPI backend for ML Cashflow Allocation and Gig Recovery.
 */

// Global Chart References
let allocationChart = null;
let scatterChart = null;
let scoresBarChart = null;
let currentTopJobs = [];
let currentAllJobs = [];

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  initDefaults();
  initEventListeners();
  loadBenchmarks();
  
  // Initial run with Demo 1
  runAnalysis();
});

/**
 * Sets initial dates and defaults
 */
function initDefaults() {
  const dateInput = document.getElementById("inputDate");
  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, "0");
  const dd = String(today.getDate()).padStart(2, "0");
  dateInput.value = `${yyyy}-${mm}-${dd}`;

  updateDaysRemainingDisplay(today);
  lucide.createIcons();
}

/**
 * Calculates days remaining in month inclusive
 */
function updateDaysRemainingDisplay(currentDate) {
  const y = currentDate.getFullYear();
  const m = currentDate.getMonth();
  const lastDay = new Date(y, m + 1, 0).getDate();
  const remaining = (lastDay - currentDate.getDate()) + 1;
  const hintEl = document.getElementById("daysRemainingHint");
  if (hintEl) {
    hintEl.textContent = `${remaining} day(s) left in month`;
  }
}

/**
 * Event Listeners
 */
function initEventListeners() {
  // Date change
  const dateInput = document.getElementById("inputDate");
  dateInput.addEventListener("change", (e) => {
    if (e.target.value) {
      updateDaysRemainingDisplay(new Date(e.target.value));
    }
  });

  // Radius slider
  const radiusSlider = document.getElementById("inputRadius");
  const radiusVal = document.getElementById("radiusVal");
  radiusSlider.addEventListener("input", (e) => {
    radiusVal.textContent = `${parseFloat(e.target.value).toFixed(1)} km`;
  });

  // Form submit
  const form = document.getElementById("ledgerForm");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runAnalysis();
  });

  // Demo 1 Preset (Earning Day: ₹1,800, Safe Surplus Day)
  document.getElementById("btnDemo1").addEventListener("click", () => {
    applyPreset({
      earnings: 1800,
      basics: 350,
      rent: 8000,
      groceries: 5500,
      emis: 3000,
      reserved: 21000,
      bills: 2200,
      family: 3,
      skills: "electrical wiring, appliance repair, plumbing",
      shift: "Evening",
      radius: 5.0,
      forceRecover: false
    });
  });

  // Demo 2 Preset (Zero Earning Day: ₹0 Alert -> Triggers Recovery)
  document.getElementById("btnDemo2").addEventListener("click", () => {
    applyPreset({
      earnings: 0,
      basics: 350,
      rent: 8000,
      groceries: 5500,
      emis: 3000,
      reserved: 18000,
      bills: 2200,
      family: 3,
      skills: "electrical wiring, appliance repair, plumbing",
      shift: "Evening",
      radius: 5.0,
      forceRecover: false
    });
  });

  // Benchmark Modal Triggers
  const benchmarkModal = document.getElementById("benchmarkModal");
  document.getElementById("btnOpenBenchmark").addEventListener("click", () => {
    benchmarkModal.classList.remove("hidden");
  });
  document.getElementById("btnCloseBenchmark").addEventListener("click", () => {
    benchmarkModal.classList.add("hidden");
  });
  document.getElementById("btnCloseBenchmark2").addEventListener("click", () => {
    benchmarkModal.classList.add("hidden");
  });

  // Gig Acceptance Modal Triggers
  const gigModal = document.getElementById("gigModal");
  document.getElementById("btnCloseGigModal").addEventListener("click", () => {
    gigModal.classList.add("hidden");
  });
  document.getElementById("btnCloseGigModal2").addEventListener("click", () => {
    gigModal.classList.add("hidden");
  });
  document.getElementById("btnCallHirer").addEventListener("click", () => {
    alert("Initiating secure worker-hirer call line...\nConnecting +91 98450 12894");
    gigModal.classList.add("hidden");
  });
}

/**
 * Apply Preset values
 */
function applyPreset(data) {
  document.getElementById("inputEarnings").value = data.earnings;
  document.getElementById("inputBasics").value = data.basics;
  document.getElementById("inputRent").value = data.rent;
  document.getElementById("inputGroceries").value = data.groceries;
  document.getElementById("inputEmis").value = data.emis;
  document.getElementById("inputReserved").value = data.reserved;
  document.getElementById("inputBills").value = data.bills;
  document.getElementById("inputFamily").value = data.family;
  document.getElementById("inputSkills").value = data.skills;
  document.getElementById("selectShift").value = data.shift;
  document.getElementById("inputRadius").value = data.radius;
  document.getElementById("radiusVal").textContent = `${data.radius.toFixed(1)} km`;
  document.getElementById("forceRecoverToggle").checked = data.forceRecover;

  runAnalysis();
}

/**
 * Executes ML API call to /api/analyze
 */
async function runAnalysis() {
  const submitBtn = document.getElementById("btnSubmit");
  const originalBtnHtml = submitBtn.innerHTML;
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="inline-block animate-spin mr-2">⏳</span> Computing ML Allocation...`;

  const payload = {
    current_date: document.getElementById("inputDate").value,
    todays_earnings: parseFloat(document.getElementById("inputEarnings").value) || 0,
    monthly_rent: parseFloat(document.getElementById("inputRent").value) || 0,
    monthly_groceries: parseFloat(document.getElementById("inputGroceries").value) || 0,
    monthly_emis: parseFloat(document.getElementById("inputEmis").value) || 0,
    already_reserved: parseFloat(document.getElementById("inputReserved").value) || 0,
    approaching_bills: parseFloat(document.getElementById("inputBills").value) || 0,
    todays_basic_expenses: parseFloat(document.getElementById("inputBasics").value) || 0,
    family_members: parseInt(document.getElementById("inputFamily").value) || 3,
    worker_skills: document.getElementById("inputSkills").value,
    preferred_shift: document.getElementById("selectShift").value,
    max_distance_km: parseFloat(document.getElementById("inputRadius").value) || 5.0,
    force_recover: document.getElementById("forceRecoverToggle").checked
  };

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    renderProtectResults(data.protect, data.meta);
    renderRecoverResults(data.recover, data.protect.shortage);

  } catch (err) {
    console.error("Analysis Error:", err);
    alert(`Error running ML Engine: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalBtnHtml;
    lucide.createIcons();
  }
}

/**
 * Render Feature 1: PROTECT results
 */
function renderProtectResults(protect, meta) {
  // Hero KPI updates
  document.getElementById("kpiSavingsPct").textContent = `${protect.savings_percentage}%`;
  document.getElementById("kpiAmountToSave").textContent = `₹${protect.amount_to_save_today.toLocaleString("en-IN")}`;
  document.getElementById("kpiSafeToSpend").textContent = `₹${protect.safe_to_spend.toLocaleString("en-IN")}`;
  document.getElementById("kpiDaysRemaining").textContent = meta.days_remaining;

  // Status Badge
  const statusBadge = document.getElementById("protectStatusBadge");
  statusBadge.textContent = protect.status_badge;
  statusBadge.className = "badge-pill text-xs";
  if (protect.status_color === "green") {
    statusBadge.classList.add("badge-green");
  } else if (protect.status_color === "amber") {
    statusBadge.classList.add("badge-amber");
  } else {
    statusBadge.classList.add("badge-red");
  }

  // Status message
  document.getElementById("protectStatusMessage").textContent = protect.status_message;

  // Financial Ledger Table
  document.getElementById("tblPredictedTotal").textContent = `₹${protect.predicted_monthly_expenditure.toLocaleString("en-IN")}`;
  document.getElementById("tblAlreadyReserved").textContent = `₹${protect.already_reserved.toLocaleString("en-IN")}`;
  document.getElementById("tblRemainingNeeded").textContent = `₹${protect.remaining_target_needed.toLocaleString("en-IN")}`;
  document.getElementById("tblDailyTarget").textContent = `₹${protect.required_daily_reserve.toLocaleString("en-IN")} / day`;

  const tblShortage = document.getElementById("tblTodayShortage");
  if (protect.shortage > 0) {
    tblShortage.textContent = `₹${protect.shortage.toLocaleString("en-IN")} (Deficit)`;
    tblShortage.className = "px-3.5 py-2 text-right font-mono text-red-600 font-bold";
  } else {
    tblShortage.textContent = `₹0 (Fully Funded)`;
    tblShortage.className = "px-3.5 py-2 text-right font-mono text-emerald-600 font-bold";
  }

  // Update Donut Chart
  renderAllocationChart(protect.chart_breakdown, protect.todays_earnings);
}

/**
 * Render Allocation Donut Chart
 */
function renderAllocationChart(breakdown, totalEarnings) {
  document.getElementById("chartCenterText").textContent = `₹${totalEarnings.toLocaleString("en-IN")}`;

  const ctx = document.getElementById("allocationChart").getContext("2d");
  const dataVals = [
    breakdown.basic_needs,
    breakdown.locked_reserve,
    breakdown.safe_to_spend
  ];

  if (allocationChart) {
    allocationChart.destroy();
  }

  allocationChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Daily Basics", "Monthly Reserve", "Safe to Spend"],
      datasets: [{
        data: dataVals.every(v => v === 0) ? [1, 0, 0] : dataVals,
        backgroundColor: dataVals.every(v => v === 0) 
          ? ["#E2E8F0", "#E2E8F0", "#E2E8F0"]
          : ["#64748B", "#059669", "#2563EB"],
        borderWidth: 2,
        borderColor: "#FFFFFF",
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "74%",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const val = context.raw || 0;
              return ` ${context.label}: ₹${val.toLocaleString("en-IN")}`;
            }
          }
        }
      }
    }
  });
}

/**
 * Render Feature 2: RECOVER results
 */
function renderRecoverResults(recover, shortage) {
  const alertBanner = document.getElementById("recoveryAlertBanner");
  const alertGap = document.getElementById("alertGapAmount");
  const alertReason = document.getElementById("alertTriggerReason");
  const recoverActivePill = document.getElementById("recoverActivePill");

  currentTopJobs = recover.top_jobs || [];
  currentAllJobs = recover.all_jobs || [];

  if (recover.triggered) {
    alertBanner.classList.remove("hidden");
    alertGap.textContent = `₹${shortage.toLocaleString("en-IN")}`;
    alertReason.textContent = recover.reason;
    recoverActivePill.textContent = "Triggered (Active)";
    recoverActivePill.className = "badge-pill badge-red text-xs";
  } else {
    alertBanner.classList.add("hidden");
    recoverActivePill.textContent = "Idle (Safe Cashflow)";
    recoverActivePill.className = "badge-pill badge-green text-xs";
  }

  // Populate Top 5 Job Cards
  const container = document.getElementById("topJobsContainer");
  container.innerHTML = "";

  if (currentTopJobs.length === 0) {
    container.innerHTML = `
      <div class="p-6 text-center text-slate-400 bg-white rounded-xl border border-slate-200">
        <p>No job matches found. Expand distance radius or adjust skill keywords.</p>
      </div>
    `;
    return;
  }

  currentTopJobs.forEach((job, index) => {
    const card = document.createElement("div");
    card.className = "fintech-card p-4 bg-white flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 hover:border-blue-400";

    const tagsHtml = (job.matched_tags || [])
      .map(t => `<span class="bg-blue-50 text-blue-700 border border-blue-200/60 text-[10px] px-2 py-0.5 rounded-full font-medium">#${t}</span>`)
      .join(" ");

    card.innerHTML = `
      <div class="flex items-start gap-3 w-full sm:w-auto">
        <div class="w-8 h-8 rounded-lg bg-slate-900 text-white font-mono font-bold flex items-center justify-center text-xs shrink-0 mt-0.5">
          #${index + 1}
        </div>
        <div class="space-y-1">
          <div class="flex items-center gap-2 flex-wrap">
            <h4 class="text-xs sm:text-sm font-bold text-slate-900">${job.title}</h4>
            <span class="badge-pill badge-green text-[10px] py-0.5 px-2">
              <i data-lucide="check" class="w-3 h-3"></i> ${job.ml_match_score.toFixed(1)}% Match
            </span>
          </div>
          <div class="flex items-center gap-3 text-slate-500 text-xs flex-wrap font-medium">
            <span class="flex items-center gap-1">
              <i data-lucide="map-pin" class="w-3.5 h-3.5 text-blue-600"></i> ${job.distance_km} km (${job.location})
            </span>
            <span>•</span>
            <span class="flex items-center gap-1">
              <i data-lucide="clock" class="w-3.5 h-3.5 text-slate-400"></i> ${job.shift} Shift (${job.duration_hrs} hrs)
            </span>
          </div>
          <div class="pt-1 flex items-center gap-1 flex-wrap">
            ${tagsHtml}
          </div>
        </div>
      </div>

      <div class="flex items-center justify-between sm:justify-end gap-3 w-full sm:w-auto pt-2 sm:pt-0 border-t sm:border-t-0 border-slate-100">
        <div class="text-left sm:text-right">
          <div class="text-base sm:text-lg font-black text-emerald-700 font-mono">₹${job.payout_inr}</div>
          <div class="text-[10px] text-slate-400 font-mono">₹${Math.round(job.payout_inr / job.duration_hrs)}/hr</div>
        </div>
        <button class="btn-primary text-xs py-2 px-3 shrink-0" onclick="openGigModal('${job.job_id}')">
          <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>
          <span>Apply</span>
        </button>
      </div>
    `;

    container.appendChild(card);
  });

  // Render Charts
  renderScatterChart(currentTopJobs, currentAllJobs, shortage);
  renderScoresBarChart(currentTopJobs);
}

/**
 * Open Gig Acceptance Modal
 */
window.openGigModal = function(jobId) {
  const job = currentAllJobs.find(j => j.job_id === jobId) || currentTopJobs.find(j => j.job_id === jobId);
  if (!job) return;

  document.getElementById("modalJobId").textContent = job.job_id;
  document.getElementById("modalJobTitle").textContent = job.title;
  document.getElementById("modalJobLocation").textContent = `${job.location} (${job.distance_km} km away)`;
  document.getElementById("modalPayout").textContent = `₹${job.payout_inr}`;
  document.getElementById("modalShift").textContent = `${job.shift} (${job.duration_hrs} hrs)`;

  document.getElementById("gigModal").classList.remove("hidden");
  lucide.createIcons();
};

/**
 * Render ML Decision Space Scatter Chart
 * Distance vs Payout across all 30 jobs
 */
function renderScatterChart(topJobs, allJobs, shortageAmount) {
  const ctx = document.getElementById("scatterChart").getContext("2d");
  const topJobIds = new Set(topJobs.map(j => j.job_id));

  const topPoints = topJobs.map(j => ({
    x: j.distance_km,
    y: j.payout_inr,
    title: j.title,
    jobId: j.job_id,
    shift: j.shift,
    score: j.ml_match_score.toFixed(1)
  }));

  const otherPoints = allJobs
    .filter(j => !topJobIds.has(j.job_id))
    .map(j => ({
      x: j.distance_km,
      y: j.payout_inr,
      title: j.title,
      jobId: j.job_id,
      shift: j.shift,
      score: j.ml_match_score ? j.ml_match_score.toFixed(1) : "0"
    }));

  if (scatterChart) {
    scatterChart.destroy();
  }

  scatterChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Top 5 ML Recommended Gigs",
          data: topPoints,
          backgroundColor: "#059669",
          borderColor: "#064E3B",
          borderWidth: 2,
          pointRadius: 9,
          pointHoverRadius: 12
        },
        {
          label: "Other Available Market Gigs",
          data: otherPoints,
          backgroundColor: "#94A3B8",
          borderColor: "#64748B",
          borderWidth: 1,
          pointRadius: 5,
          pointHoverRadius: 8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: "Distance from Worker Location (km)",
            font: { weight: "bold", size: 11 }
          },
          grid: { color: "#F1F5F9" },
          min: 0,
          max: 30
        },
        y: {
          title: {
            display: true,
            text: "Estimated Job Payout (₹)",
            font: { weight: "bold", size: 11 }
          },
          grid: { color: "#F1F5F9" },
          min: 400,
          max: 2300
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: { font: { size: 11, weight: "bold" }, usePointStyle: true }
        },
        tooltip: {
          callbacks: {
            title: function(items) {
              const raw = items[0].raw;
              return `[${raw.jobId}] ${raw.title}`;
            },
            label: function(item) {
              const raw = item.raw;
              return [
                ` ML Match Score: ${raw.score}%`,
                ` Payout: ₹${raw.y.toLocaleString("en-IN")}`,
                ` Distance: ${raw.x} km`,
                ` Shift: ${raw.shift}`
              ];
            }
          }
        }
      }
    }
  });
}

/**
 * Render Top 5 Scores Bar Chart
 */
function renderScoresBarChart(topJobs) {
  const ctx = document.getElementById("scoresBarChart").getContext("2d");
  const labels = topJobs.map(j => `${j.job_id} (${j.distance_km}km)`);
  const scores = topJobs.map(j => j.ml_match_score);

  if (scoresBarChart) {
    scoresBarChart.destroy();
  }

  scoresBarChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "ML Match Score (%)",
        data: scores,
        backgroundColor: [
          "#059669",
          "#10B981",
          "#2563EB",
          "#3B82F6",
          "#60A5FA"
        ],
        borderRadius: 6,
        barPercentage: 0.6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: 0,
          max: 100,
          title: { display: true, text: "Match Confidence (%)", font: { weight: "bold", size: 10 } },
          grid: { color: "#F1F5F9" }
        },
        y: {
          grid: { display: false },
          ticks: { font: { weight: "bold", size: 10 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Confidence: ${ctx.raw.toFixed(1)}%`
          }
        }
      }
    }
  });
}

/**
 * Load ML Model Benchmarks for Judge Modal
 */
async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmark");
    const json = await res.json();
    const tbody = document.getElementById("benchmarkTableBody");
    tbody.innerHTML = "";

    const models = json.benchmark_models || {};
    for (const [name, stats] of Object.entries(models)) {
      const isWinner = name.includes("Random Forest");
      const tr = document.createElement("tr");
      tr.className = isWinner ? "bg-emerald-50/60 font-semibold text-slate-900" : "hover:bg-slate-50";

      tr.innerHTML = `
        <td class="px-4 py-2.5 flex items-center gap-2">
          ${isWinner ? '<i data-lucide="check-circle-2" class="w-4 h-4 text-emerald-600"></i>' : '<span class="w-4 inline-block"></span>'}
          <span>${name}</span>
        </td>
        <td class="px-3 py-2.5 text-right font-mono">${stats.r2_score.toFixed(4)}</td>
        <td class="px-3 py-2.5 text-right font-mono font-bold ${isWinner ? 'text-emerald-700' : 'text-slate-700'}">${stats.accuracy_pct.toFixed(2)}%</td>
        <td class="px-4 py-2.5 text-right font-mono">₹${stats.mae_inr.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        <td class="px-3 py-2.5 text-center">
          ${isWinner 
            ? '<span class="badge-pill badge-green text-[10px] py-0.5 px-2">Deployed (.pkl)</span>' 
            : '<span class="text-slate-400 text-[11px]">Benchmarked</span>'}
        </td>
      `;
      tbody.appendChild(tr);
    }
    lucide.createIcons();
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}
