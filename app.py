import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="MEVPRO-1 Phase 3 Forecaster", layout="wide")

st.title("MEVPRO-1 (mCRPC) Phase 3 Probability Forecaster")
st.markdown("### Monte Carlo Simulation & Sensitivity Dashboard")
st.markdown(
    "This tool models the full chain from Phase 1b signal through statistical "
    "testing using a 100,000-draw Monte Carlo simulation. Each adjustable parameter "
    "corresponds to a key input assumption — the drug's true effect, the trial's "
    "statistical design, and the Phase 1b → Phase 3 translation risks."
)

# ============================================================
# SIDEBAR — INPUT PARAMETERS
# ============================================================
st.sidebar.header("Phase 3 HR Adjustments")
st.sidebar.markdown("Penalties applied to the Phase 1b baseline HR of 0.51")

base_hr = 0.51

ecog = st.sidebar.slider("ECOG Imbalance Normalization", 0.0, 0.10, 0.04, 0.01,
    help="Phase 1b combo arm had 63% ECOG 0 vs 38% in control. Phase 3 stratification removes this advantage.")
reg_mean = st.sidebar.slider("Regression to Mean (Small-N)", 0.0, 0.15, 0.09, 0.01,
    help="Phase 1b HR 0.51 came from only 34 events with 90% CI 0.28-0.95. Larger samples regress toward true mean.")
dose = st.sidebar.slider("Dose Reduction (1250mg → 875mg)", 0.0, 0.05, 0.02, 0.01,
    help="PK bridging shows AUC equivalence but Cmax is 21% lower. Small residual risk.")
pop_shift = st.sidebar.slider("Population Shift (enza-naïve)", -0.05, 0.05, -0.02, 0.01,
    help="Phase 3 enrolls cleaner post-abi enza-naïve patients vs heavily pretreated Phase 1b. Slight bullish.")
docetaxel = st.sidebar.slider("Stronger Blended Control Arm", 0.0, 0.10, 0.03, 0.01,
    help="Phase 3 control includes docetaxel option (median ~8-9 mo) vs pure enza in Phase 1b (~6 mo).")
site_dilution = st.sidebar.slider("Community Site Dilution", 0.0, 0.05, 0.02, 0.01,
    help="Phase 1b at ~15 academic centers; Phase 3 at 100+ global sites including community.")
bicr = st.sidebar.slider("BICR vs Investigator Assessment", 0.0, 0.05, 0.02, 0.01,
    help="BICR is stricter than investigator assessment; eliminates open-label bias.")

st.sidebar.markdown("---")
st.sidebar.header("Trial Design Parameters")
n_events = st.sidebar.slider("Number of rPFS Events", 200, 400, 302, 1,
    help="Pfizer's protocol target: 302 events for primary analysis.")
alpha = st.sidebar.selectbox("Significance Threshold (2-sided)", [0.05, 0.025, 0.01], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Uncertainty Settings")
hr_uncertainty = st.sidebar.slider("Std Dev of True HR (Uncertainty)", 0.03, 0.12, 0.07, 0.01,
    help="How uncertain are you about the true Phase 3 HR? Higher = wider distribution.")
n_sims = st.sidebar.selectbox("Monte Carlo Draws", [10000, 50000, 100000], index=2)

# ============================================================
# CORE MATH
# ============================================================
total_penalty = ecog + reg_mean + dose + pop_shift + docetaxel + site_dilution + bicr
adj_hr = base_hr + total_penalty

# Power formula: Power ≈ Φ(√(D/4) × |ln(HR)| − Z_α/2)
def calc_power(hr, events, alpha_level):
    if hr >= 1.0:
        return alpha_level / 2
    z_alpha = stats.norm.ppf(1 - alpha_level / 2)
    z_power = np.sqrt(events / 4) * np.abs(np.log(hr)) - z_alpha
    return stats.norm.cdf(z_power)

# Power at the central HR estimate
power_central = calc_power(adj_hr, n_events, alpha)

# Monte Carlo simulation
np.random.seed(42)
hr_distribution = np.random.normal(loc=adj_hr, scale=hr_uncertainty, size=n_sims)
hr_distribution = np.clip(hr_distribution, 0.30, 1.50)

# For each simulated true HR, simulate trial outcome
power_at_each_hr = np.array([calc_power(hr, n_events, alpha) for hr in hr_distribution])
trial_wins = np.random.binomial(1, power_at_each_hr)
pos = trial_wins.mean()

# Outcome classification (based on observed HR if trial wins)
def classify_outcome(true_hr, trial_won):
    if not trial_won:
        return "Loss"
    if true_hr <= 0.65:
        return "Blowout Win"
    elif true_hr <= 0.70:
        return "Clear Win"
    elif true_hr <= 0.75:
        return "Solid Win"
    else:
        return "Marginal Win"

outcomes = [classify_outcome(hr, win) for hr, win in zip(hr_distribution, trial_wins)]
outcome_counts = pd.Series(outcomes).value_counts(normalize=True) * 100
outcome_order = ["Blowout Win", "Clear Win", "Solid Win", "Marginal Win", "Loss"]
outcome_counts = outcome_counts.reindex(outcome_order, fill_value=0)

# ============================================================
# TOP METRICS ROW
# ============================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Phase 1b Baseline HR", f"{base_hr:.2f}")
col2.metric("Total Penalty", f"+{total_penalty:.2f}")
col3.metric("Central Phase 3 HR Estimate", f"{adj_hr:.2f}")
col4.metric("Probability of Success (PoS)", f"{pos*100:.1f}%")

# ============================================================
# OUTCOME CLASSIFICATION
# ============================================================
st.markdown("---")
st.subheader("First-Read Outcome Classification")
st.markdown(f"Across **{n_sims:,}** simulated trials, here's how MEVPRO-1 reads out:")

color_map = {
    "Blowout Win": "#1a9641",
    "Clear Win": "#73c378",
    "Solid Win": "#c4e687",
    "Marginal Win": "#fdae61",
    "Loss": "#d7191c"
}
colors = [color_map[o] for o in outcome_counts.index]

fig_outcome = go.Figure(data=[
    go.Bar(
        x=outcome_counts.index,
        y=outcome_counts.values,
        text=[f"{v:.1f}%" for v in outcome_counts.values],
        textposition="outside",
        marker_color=colors
    )
])
fig_outcome.update_layout(
    yaxis=dict(title="Probability (%)", range=[0, max(outcome_counts.values) * 1.2]),
    xaxis=dict(title="Outcome"),
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    height=400
)
st.plotly_chart(fig_outcome, use_container_width=True)

win_pct = 100 - outcome_counts["Loss"]
st.markdown(f"**Total win probability: {win_pct:.1f}%** | **Loss probability: {outcome_counts['Loss']:.1f}%**")

# ============================================================
# HR DISTRIBUTION
# ============================================================
st.markdown("---")
st.subheader("Simulated Phase 3 HR Distribution")
st.markdown(
    f"Distribution of {n_sims:,} simulated true HRs with central estimate **{adj_hr:.2f}** "
    f"and uncertainty **±{hr_uncertainty:.2f}**. Vertical lines mark key thresholds."
)

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=hr_distribution,
    nbinsx=80,
    marker_color="#4575b4",
    opacity=0.75,
    name="Simulated HRs"
))
fig_dist.add_vline(x=adj_hr, line_dash="solid", line_color="black",
    annotation_text=f"Central HR ({adj_hr:.2f})", annotation_position="top")
fig_dist.add_vline(x=0.66, line_dash="dash", line_color="green",
    annotation_text="Pfizer Target (0.66)", annotation_position="top left")
fig_dist.add_vline(x=0.80, line_dash="dash", line_color="orange",
    annotation_text="Borderline (0.80)", annotation_position="top right")
fig_dist.add_vline(x=1.0, line_dash="dash", line_color="red",
    annotation_text="No Benefit (1.0)", annotation_position="top right")
fig_dist.update_layout(
    xaxis=dict(title="True Phase 3 Hazard Ratio", range=[0.3, 1.3]),
    yaxis=dict(title="Frequency"),
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    height=400
)
st.plotly_chart(fig_dist, use_container_width=True)

p_below_080 = (hr_distribution < 0.80).mean() * 100
p_below_075 = (hr_distribution < 0.75).mean() * 100
p_below_070 = (hr_distribution < 0.70).mean() * 100
st.markdown(
    f"**P(true HR < 0.70) = {p_below_070:.1f}%** | "
    f"**P(true HR < 0.75) = {p_below_075:.1f}%** | "
    f"**P(true HR < 0.80) = {p_below_080:.1f}%**"
)

# ============================================================
# WATERFALL CHART
# ============================================================
st.markdown("---")
st.subheader("HR Walk-Down: Phase 1b → Phase 3")

fig_wf = go.Figure(go.Waterfall(
    name="MEVPRO-1 HR Bridge",
    orientation="v",
    measure=["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"],
    x=["Phase 1b<br>Baseline", "ECOG", "Regression<br>to Mean", "Dose<br>Reduction",
       "Population<br>Shift", "Stronger<br>Control", "Site<br>Dilution", "BICR", "Phase 3<br>Estimate"],
    text=[f"{base_hr:.2f}", f"+{ecog:.2f}", f"+{reg_mean:.2f}", f"+{dose:.2f}",
          f"{pop_shift:+.2f}", f"+{docetaxel:.2f}", f"+{site_dilution:.2f}", f"+{bicr:.2f}", f"{adj_hr:.2f}"],
    textposition="outside",
    y=[base_hr, ecog, reg_mean, dose, pop_shift, docetaxel, site_dilution, bicr, adj_hr],
    decreasing={"marker": {"color": "#2ca02c"}},
    increasing={"marker": {"color": "#d62728"}},
    totals={"marker": {"color": "#1f77b4"}},
    connector={"line": {"color": "rgb(63, 63, 63)"}}
))
fig_wf.update_layout(
    yaxis=dict(title="Hazard Ratio", range=[0.4, 0.85]),
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    height=450
)
st.plotly_chart(fig_wf, use_container_width=True)

# ============================================================
# POWER CURVE
# ============================================================
st.markdown("---")
st.subheader("Statistical Power Curve")
st.markdown(
    f"How does the trial's probability of detecting a real effect change with the true HR? "
    f"Calculated for **{n_events} events** at α = {alpha} (two-sided)."
)

hr_range = np.linspace(0.50, 1.00, 100)
power_curve = [calc_power(hr, n_events, alpha) * 100 for hr in hr_range]

fig_power = go.Figure()
fig_power.add_trace(go.Scatter(
    x=hr_range, y=power_curve, mode='lines',
    line=dict(color='#1f77b4', width=3), name='Power'
))
fig_power.add_vline(x=adj_hr, line_dash="solid", line_color="black",
    annotation_text=f"Your central HR ({adj_hr:.2f}, Power={power_central*100:.1f}%)",
    annotation_position="top")
fig_power.add_vline(x=0.66, line_dash="dash", line_color="green",
    annotation_text="Pfizer Target", annotation_position="top left")
fig_power.add_hline(y=80, line_dash="dot", line_color="gray",
    annotation_text="80% power threshold", annotation_position="bottom right")
fig_power.update_layout(
    xaxis=dict(title="True Hazard Ratio"),
    yaxis=dict(title="Statistical Power (%)", range=[0, 105]),
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    height=400
)
st.plotly_chart(fig_power, use_container_width=True)

# ============================================================
# SENSITIVITY TORNADO
# ============================================================
st.markdown("---")
st.subheader("Sensitivity Analysis (Tornado Chart)")
st.markdown(
    "Which input assumptions move the PoS the most? Each bar shows the PoS impact "
    "of moving each parameter from its low to high value while holding others at base."
)

base_params = {
    "ECOG Imbalance": ecog,
    "Regression to Mean": reg_mean,
    "Dose Reduction": dose,
    "Population Shift": pop_shift,
    "Stronger Control": docetaxel,
    "Site Dilution": site_dilution,
    "BICR Penalty": bicr
}
ranges = {
    "ECOG Imbalance": (0.0, 0.10),
    "Regression to Mean": (0.0, 0.15),
    "Dose Reduction": (0.0, 0.05),
    "Population Shift": (-0.05, 0.05),
    "Stronger Control": (0.0, 0.10),
    "Site Dilution": (0.0, 0.05),
    "BICR Penalty": (0.0, 0.05)
}

def calc_pos(params):
    hr = base_hr + sum(params.values())
    sims = np.random.normal(loc=hr, scale=hr_uncertainty, size=10000)
    sims = np.clip(sims, 0.30, 1.50)
    powers = np.array([calc_power(h, n_events, alpha) for h in sims])
    return np.random.binomial(1, powers).mean()

sensitivity = []
for param, base_val in base_params.items():
    low_params = base_params.copy()
    high_params = base_params.copy()
    low_params[param] = ranges[param][0]
    high_params[param] = ranges[param][1]
    pos_low = calc_pos(low_params)
    pos_high = calc_pos(high_params)
    sensitivity.append({
        "Parameter": param,
        "Low PoS": pos_low * 100,
        "High PoS": pos_high * 100,
        "Swing": abs(pos_high - pos_low) * 100
    })

sens_df = pd.DataFrame(sensitivity).sort_values("Swing", ascending=True)

fig_tornado = go.Figure()
fig_tornado.add_trace(go.Bar(
    y=sens_df["Parameter"],
    x=sens_df["Low PoS"] - pos * 100,
    orientation='h',
    name='Bullish (low penalty)',
    marker_color='#2ca02c',
    text=[f"{v:.1f}%" for v in sens_df["Low PoS"]],
    textposition="outside"
))
fig_tornado.add_trace(go.Bar(
    y=sens_df["Parameter"],
    x=sens_df["High PoS"] - pos * 100,
    orientation='h',
    name='Bearish (high penalty)',
    marker_color='#d62728',
    text=[f"{v:.1f}%" for v in sens_df["High PoS"]],
    textposition="outside"
))
fig_tornado.update_layout(
    barmode='overlay',
    xaxis=dict(title=f"PoS Change from Base ({pos*100:.1f}%)"),
    plot_bgcolor='rgba(0,0,0,0)',
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_tornado, use_container_width=True)

# ============================================================
# HR-TO-MEDIAN RPFS TRANSLATION
# ============================================================
st.markdown("---")
st.subheader("HR → Median rPFS Translation")
st.markdown(
    "Using HR ≈ Control Median / Experimental Median, here's what your central HR "
    "implies for the Phase 3 rPFS curves (assuming a 6.75-month control median per Pfizer's protocol)."
)

control_median = 6.75
exp_median = control_median / adj_hr
delta = exp_median - control_median

col_a, col_b, col_c = st.columns(3)
col_a.metric("Implied Control Median rPFS", f"{control_median:.2f} mo")
col_b.metric("Implied Experimental Median rPFS", f"{exp_median:.2f} mo")
col
