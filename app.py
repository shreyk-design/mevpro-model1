import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from math import erf, sqrt, log

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="MEVPRO-1 Forecaster",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for warpspeed-style dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d3548;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label {
        color: #8b92a8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
    }
    .metric-delta-positive {
        color: #2ecc71;
        font-size: 14px;
    }
    .metric-delta-negative {
        color: #e74c3c;
        font-size: 14px;
    }
    h1 {
        color: #ffffff !important;
        font-weight: 700;
    }
    h2, h3 {
        color: #e8eaed !important;
    }
    .scenario-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("# MEVPRO-1 Phase 3 Probability Forecaster")
st.markdown("##### Pfizer mevrometostat (PF-06821497) + enzalutamide vs physician's choice in post-abi mCRPC")
st.markdown(
    "<p style='color:#8b92a8;'>Monte Carlo simulation modeling the full chain from Phase 1b signal "
    "through Phase 3 statistical testing. Adjust parameters in the sidebar to stress-test assumptions.</p>",
    unsafe_allow_html=True
)

# ============================================================
# HELPER FUNCTIONS (NO SCIPY DEPENDENCY)
# ============================================================
def norm_cdf(x):
    """Standard normal CDF using error function."""
    return 0.5 * (1 + erf(x / sqrt(2)))

def norm_ppf(p):
    """Inverse standard normal CDF (approximation)."""
    # Beasley-Springer-Moro approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

def calc_power(hr, events, alpha_level=0.05):
    """Log-rank power formula for 1:1 randomized design."""
    if hr >= 1.0:
        return alpha_level / 2
    if hr <= 0:
        return 1.0
    z_alpha = norm_ppf(1 - alpha_level / 2)
    z_power = sqrt(events / 4) * abs(log(hr)) - z_alpha
    return norm_cdf(z_power)

def run_monte_carlo(central_hr, uncertainty, events, alpha, n_sims=50000, seed=42):
    """Run Monte Carlo simulation."""
    np.random.seed(seed)
    hr_dist = np.random.normal(loc=central_hr, scale=uncertainty, size=n_sims)
    hr_dist = np.clip(hr_dist, 0.30, 1.50)
    powers = np.array([calc_power(hr, events, alpha) for hr in hr_dist])
    wins = np.random.binomial(1, powers)
    return hr_dist, powers, wins

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## Phase 3 HR Adjustments")
st.sidebar.caption("Penalties applied to the Phase 1b baseline HR of 0.51")

base_hr = 0.51

ecog = st.sidebar.slider("ECOG Imbalance", 0.0, 0.10, 0.04, 0.01,
    help="Phase 1b combo arm had 63% ECOG 0 vs 38% in control. Phase 3 stratification removes this advantage.")
reg_mean = st.sidebar.slider("Regression to Mean", 0.0, 0.15, 0.09, 0.01,
    help="Phase 1b HR 0.51 came from only 34 events with 90% CI 0.28-0.95. Larger samples regress toward true mean.")
dose = st.sidebar.slider("Dose Reduction", 0.0, 0.05, 0.02, 0.01,
    help="PK bridging shows AUC equivalence but Cmax is 21% lower.")
pop_shift = st.sidebar.slider("Population Shift", -0.05, 0.05, -0.02, 0.01,
    help="Phase 3 enrolls cleaner post-abi enza-naïve patients.")
docetaxel = st.sidebar.slider("Stronger Blended Control", 0.0, 0.10, 0.03, 0.01,
    help="Phase 3 control includes docetaxel option (~8-9 mo) vs pure enza in Phase 1b.")
site_dilution = st.sidebar.slider("Site Dilution", 0.0, 0.05, 0.02, 0.01,
    help="Phase 1b at ~15 academic centers; Phase 3 at 100+ global sites.")
bicr = st.sidebar.slider("BICR vs Investigator", 0.0, 0.05, 0.02, 0.01,
    help="BICR is stricter than investigator assessment.")

st.sidebar.markdown("---")
st.sidebar.markdown("## Trial Design")
n_events = st.sidebar.slider("rPFS Events", 200, 400, 302, 1,
    help="Pfizer's protocol target: 302 events.")
alpha = st.sidebar.selectbox("Significance Threshold", [0.05, 0.025, 0.01], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## Uncertainty")
hr_uncertainty = st.sidebar.slider("HR Std Dev", 0.03, 0.15, 0.07, 0.01,
    help="How wide is your prior on the true HR?")

# ============================================================
# CORE CALCULATIONS
# ============================================================
total_penalty = ecog + reg_mean + dose + pop_shift + docetaxel + site_dilution + bicr
adj_hr = base_hr + total_penalty

# Run Monte Carlo
hr_dist, powers, wins = run_monte_carlo(adj_hr, hr_uncertainty, n_events, alpha)
pos = wins.mean()
power_central = calc_power(adj_hr, n_events, alpha)

# 95% CI on observed HR (approximation)
se_logHR = 2 / sqrt(n_events)
ci_lower = adj_hr * np.exp(-1.96 * se_logHR)
ci_upper = adj_hr * np.exp(1.96 * se_logHR)

# Outcome classification
def classify_outcome(hr, won):
    if not won:
        return "Loss"
    if hr <= 0.65:
        return "Blowout Win"
    elif hr <= 0.70:
        return "Clear Win"
    elif hr <= 0.75:
        return "Solid Win"
    else:
        return "Marginal Win"

outcomes = [classify_outcome(hr, win) for hr, win in zip(hr_dist, wins)]
outcome_counts = pd.Series(outcomes).value_counts(normalize=True) * 100
outcome_order = ["Blowout Win", "Clear Win", "Solid Win", "Marginal Win", "Loss"]
outcome_counts = outcome_counts.reindex(outcome_order, fill_value=0)

# ============================================================
# TOP METRICS - DARK CARD STYLE
# ============================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Phase 1b Baseline HR</div>
        <div class="metric-value">{base_hr:.2f}</div>
        <div style="color:#8b92a8; font-size:12px;">From randomized Part 2B</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Penalty</div>
        <div class="metric-value">+{total_penalty:.2f}</div>
        <div style="color:#8b92a8; font-size:12px;">Phase 1b → Phase 3 adjustments</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Central HR Estimate</div>
        <div class="metric-value">{adj_hr:.2f}</div>
        <div style="color:#8b92a8; font-size:12px;">95% CI: {ci_lower:.2f} – {ci_upper:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    color = "#2ecc71" if pos > 0.65 else "#f39c12" if pos > 0.45 else "#e74c3c"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Probability of Success</div>
        <div class="metric-value" style="color:{color};">{pos*100:.1f}%</div>
        <div style="color:#8b92a8; font-size:12px;">Across {len(wins):,} simulations</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SCENARIO COMPARISON - BASE / UPSIDE / DOWNSIDE
# ============================================================
st.markdown("---")
st.markdown("## Scenario Analysis")
st.caption("Three pre-built scenarios alongside your custom slider configuration.")

# Define scenarios
scenarios = {
    "Downside": {"ecog": 0.07, "reg_mean": 0.13, "dose": 0.04, "pop_shift": 0.02,
                 "docetaxel": 0.07, "site_dilution": 0.04, "bicr": 0.03,
                 "color": "#e74c3c", "description": "Aggressive penalties; correlated downside risks materialize."},
    "Base": {"ecog": 0.04, "reg_mean": 0.09, "dose": 0.02, "pop_shift": -0.02,
             "docetaxel": 0.03, "site_dilution": 0.02, "bicr": 0.02,
             "color": "#3498db", "description": "Central estimate; balanced view of Phase 1b → Phase 3 translation."},
    "Upside": {"ecog": 0.02, "reg_mean": 0.05, "dose": 0.00, "pop_shift": -0.04,
               "docetaxel": 0.01, "site_dilution": 0.01, "bicr": 0.01,
               "color": "#2ecc71", "description": "Minimal attenuation; Phase 1b signal mostly preserved."}
}

scenario_results = {}
for name, params in scenarios.items():
    s_hr = base_hr + params["ecog"] + params["reg_mean"] + params["dose"] + \
           params["pop_shift"] + params["docetaxel"] + params["site_dilution"] + params["bicr"]
    s_dist, s_powers, s_wins = run_monte_carlo(s_hr, hr_uncertainty, n_events, alpha, seed=hash(name) % 100000)
    s_pos = s_wins.mean()
    scenario_results[name] = {"hr": s_hr, "pos": s_pos, "color": params["color"], "description": params["description"]}

# Add custom scenario
scenario_results["Your Custom"] = {"hr": adj_hr, "pos": pos, "color": "#9b59b6",
                                    "description": "Reflects your current slider settings."}

# Display scenario cards
sc1, sc2, sc3, sc4 = st.columns(4)
for col, (name, result) in zip([sc1, sc2, sc3, sc4], scenario_results.items()):
    with col:
        st.markdown(f"""
        <div class="scenario-box" style="border-left-color: {result['color']};">
            <div style="color:{result['color']}; font-weight:700; font-size:14px; text-transform:uppercase; letter-spacing:1px;">{name}</div>
            <div style="color:white; font-size:28px; font-weight:700; margin:8px 0;">{result['pos']*100:.1f}%</div>
            <div style="color:#8b92a8; font-size:13px;">HR: <span style='color:white;'>{result['hr']:.2f}</span></div>
            <div style="color:#8b92a8; font-size:11px; margin-top:8px;">{result['description']}</div>
        </div>
        """, unsafe_allow_html=True)

# Scenario comparison chart
fig_scenario = go.Figure()
sc_names = list(scenario_results.keys())
sc_pos = [scenario_results[n]["pos"] * 100 for n in sc_names]
sc_colors = [scenario_results[n]["color"] for n in sc_names]

fig_scenario.add_trace(go.Bar(
    x=sc_names, y=sc_pos,
    marker=dict(color=sc_colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
    text=[f"{p:.1f}%" for p in sc_pos],
    textposition="outside",
    textfont=dict(color='white', size=14)
))
fig_scenario.update_layout(
    title=dict(text="Probability of Success by Scenario", font=dict(color='white', size=18)),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    yaxis=dict(title="PoS (%)", range=[0, 100], gridcolor='#2d3548'),
    xaxis=dict(gridcolor='#2d3548'),
    showlegend=False,
    height=400
)
st.plotly_chart(fig_scenario, use_container_width=True)

# ============================================================
# OUTCOME CLASSIFICATION CHART
# ============================================================
st.markdown("---")
st.markdown("## First-Read Outcome Distribution")
st.caption(f"Across {len(wins):,} simulated trials, here's how MEVPRO-1 reads out at first analysis.")

color_map = {
    "Blowout Win": "#1a9641",
    "Clear Win": "#73c378",
    "Solid Win": "#a6d96a",
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
        textfont=dict(color='white', size=14, family="Arial Black"),
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.3)', width=1))
    )
])
fig_outcome.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    yaxis=dict(title="Probability (%)", range=[0, max(outcome_counts.values) * 1.25], gridcolor='#2d3548'),
    xaxis=dict(title="Outcome", gridcolor='#2d3548'),
    showlegend=False,
    height=450
)
st.plotly_chart(fig_outcome, use_container_width=True)

win_pct = 100 - outcome_counts["Loss"]
st.markdown(f"""
<div style="background:#1a1f2e; padding:15px; border-radius:8px; border-left:4px solid #2ecc71;">
    <span style="color:#2ecc71; font-weight:700;">Total Win Probability: {win_pct:.1f}%</span>
    <span style="color:#8b92a8; margin:0 15px;">|</span>
    <span style="color:#e74c3c; font-weight:700;">Loss Probability: {outcome_counts['Loss']:.1f}%</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HR DISTRIBUTION DENSITY
# ============================================================
st.markdown("---")
st.markdown("## Simulated Phase 3 HR Distribution")
st.caption(f"Distribution of {len(hr_dist):,} simulated true HRs. Vertical lines mark key decision thresholds.")

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=hr_dist,
    nbinsx=80,
    marker=dict(color="#4a90e2", line=dict(color='rgba(255,255,255,0.1)', width=0.5)),
    opacity=0.8,
    name="Simulated HRs"
))
fig_dist.add_vline(x=adj_hr, line_dash="solid", line_color="white", line_width=2,
    annotation_text=f"Central: {adj_hr:.2f}", annotation_position="top",
    annotation_font_color="white")
fig_dist.add_vline(x=0.66, line_dash="dash", line_color="#2ecc71", line_width=2,
    annotation_text="Pfizer Target (0.66)", annotation_position="top left",
    annotation_font_color="#2ecc71")
fig_dist.add_vline(x=0.80, line_dash="dash", line_color="#f39c12", line_width=2,
    annotation_text="Borderline (0.80)", annotation_position="top right",
    annotation_font_color="#f39c12")
fig_dist.add_vline(x=1.0, line_dash="dash", line_color="#e74c3c", line_width=2,
    annotation_text="No Benefit (1.0)", annotation_position="top right",
    annotation_font_color="#e74c3c")
fig_dist.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    xaxis=dict(title="True Phase 3 Hazard Ratio", range=[0.3, 1.3], gridcolor='#2d3548'),
    yaxis=dict(title="Frequency", gridcolor='#2d3548'),
    showlegend=False,
    height=450
)
st.plotly_chart(fig_dist, use_container_width=True)

p_below_080 = (hr_dist < 0.80).mean() * 100
p_below_075 = (hr_dist < 0.75).mean() * 100
p_below_070 = (hr_dist < 0.70).mean() * 100

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">P(True HR < 0.70)</div>
        <div class="metric-value" style="color:#73c378;">{p_below_070:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with p_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">P(True HR < 0.75)</div>
        <div class="metric-value" style="color:#fdae61;">{p_below_075:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with p_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">P(True HR < 0.80)</div>
        <div class="metric-value" style="color:#f39c12;">{p_below_080:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# WATERFALL CHART
# ============================================================
st.markdown("---")
st.markdown("## HR Walk-Down: Phase 1b → Phase 3")
st.caption("How each adjustment penalizes the Phase 1b HR of 0.51 to arrive at the Phase 3 estimate.")

fig_wf = go.Figure(go.Waterfall(
    name="HR Bridge",
    orientation="v",
    measure=["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"],
    x=["Phase 1b<br>Baseline", "ECOG", "Regression<br>to Mean", "Dose<br>Reduction",
       "Population<br>Shift", "Stronger<br>Control", "Site<br>Dilution", "BICR", "Phase 3<br>Estimate"],
    text=[f"{base_hr:.2f}", f"+{ecog:.2f}", f"+{reg_mean:.2f}", f"+{dose:.2f}",
          f"{pop_shift:+.2f}", f"+{docetaxel:.2f}", f"+{site_dilution:.2f}", f"+{bicr:.2f}", f"{adj_hr:.2f}"],
    textposition="outside",
    textfont=dict(color='white', size=12),
    y=[base_hr, ecog, reg_mean, dose, pop_shift, docetaxel, site_dilution, bicr, adj_hr],
    decreasing={"marker": {"color": "#2ecc71"}},
    increasing={"marker": {"color": "#e74c3c"}},
    totals={"marker": {"color": "#4a90e2"}},
    connector={"line": {"color": "rgba(255,255,255,0.3)"}}
))
fig_wf.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    yaxis=dict(title="Hazard Ratio", range=[0.4, 0.85], gridcolor='#2d3548'),
    xaxis=dict(gridcolor='#2d3548'),
    showlegend=False,
    height=500
)
st.plotly_chart(fig_wf, use_container_width=True)

# ============================================================
# POWER CURVE
# ============================================================
st.markdown("---")
st.markdown("## Statistical Power Curve")
st.caption(f"Trial's probability of detecting a real effect at each true HR. Calculated for {n_events} events at α = {alpha}.")

hr_range = np.linspace(0.50, 1.00, 100)
power_curve = [calc_power(hr, n_events, alpha) * 100 for hr in hr_range]

fig_power = go.Figure()
fig_power.add_trace(go.Scatter(
    x=hr_range, y=power_curve, mode='lines',
    line=dict(color='#4a90e2', width=4), name='Power',
    fill='tozeroy', fillcolor='rgba(74,144,226,0.2)'
))
fig_power.add_vline(x=adj_hr, line_dash="solid", line_color="white", line_width=2,
    annotation_text=f"Central HR ({adj_hr:.2f}) → Power {power_central*100:.1f}%",
    annotation_position="top", annotation_font_color="white")
fig_power.add_vline(x=0.66, line_dash="dash", line_color="#2ecc71", line_width=2,
    annotation_text="Pfizer Target", annotation_position="top left", annotation_font_color="#2ecc71")
fig_power.add_hline(y=80, line_dash="dot", line_color="rgba(255,255,255,0.4)",
    annotation_text="80% threshold", annotation_position="bottom right", annotation_font_color="white")
fig_power.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    xaxis=dict(title="True Hazard Ratio", gridcolor='#2d3548'),
    yaxis=dict(title="Statistical Power (%)", range=[0, 105], gridcolor='#2d3548'),
    showlegend=False,
    height=450
)
st.plotly_chart(fig_power, use_container_width=True)

# ============================================================
# SENSITIVITY TORNADO
# ============================================================
st.markdown("---")
st.markdown("## Sensitivity Tornado")
st.caption("Which input assumptions move the PoS the most? Each bar shows PoS impact of moving each parameter from low to high.")

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

def calc_pos_quick(params):
    hr = base_hr + sum(params.values())
    sims = np.random.normal(loc=hr, scale=hr_uncertainty, size=10000)
    sims = np.clip(sims, 0.30, 1.50)
    powers_arr = np.array([calc_power(h, n_events, alpha) for h in sims])
    return np.random.binomial(1, powers_arr).mean()

sensitivity = []
for param, base_val in base_params.items():
    low_p = base_params.copy()
    high_p = base_params.copy()
    low_p[param] = ranges[param][0]
    high_p[param] = ranges[param][1]
    pos_low = calc_pos_quick(low_p) * 100
    pos_high = calc_pos_quick(high_p) * 100
    sensitivity.append({
        "Parameter": param, "Low": pos_low, "High": pos_high,
        "Swing": abs(pos_high - pos_low)
    })

sens_df = pd.DataFrame(sensitivity).sort_values("Swing", ascending=True)

fig_tornado = go.Figure()
fig_tornado.add_trace(go.Bar(
    y=sens_df["Parameter"], x=sens_df["Low"] - pos * 100,
    orientation='h', name='Bullish (low penalty)',
    marker=dict(color='#2ecc71'),
    text=[f"{v:.1f}%" for v in sens_df["Low"]], textposition="outside",
    textfont=dict(color='white')
))
fig_tornado.add_trace(go.Bar(
    y=sens_df["Parameter"], x=sens_df["High"] - pos * 100,
    orientation='h', name='Bearish (high penalty)',
    marker=dict(color='#e74c3c'),
    text=[f"{v:.1f}%" for v in sens_df["High"]], textposition="outside",
    textfont=dict(color='white')
))
fig_tornado.update_layout(
    barmode='overlay',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    xaxis=dict(title=f"PoS Change vs Base ({pos*100:.1f}%)", gridcolor='#2d3548'),
    yaxis=dict(gridcolor='#2d3548'),
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white'))
)
st.plotly_chart(fig_tornado, use_container_width=True)

# ============================================================
# HR → MEDIAN RPFS TRANSLATION
# ============================================================
st.markdown("---")
st.markdown("## HR → Median rPFS Translation")
st.caption("Using HR ≈ Control Median / Experimental Median, with Pfizer's assumed 6.75-month control median.")

control_median = 6.75
exp_median = control_median / adj_hr
delta = exp_median - control_median

t1, t2, t3 = st.columns(3)
with t1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Control Median rPFS</div>
        <div class="metric-value">{control_median:.2f} mo</div>
        <div style="color:#8b92a8; font-size:12px;">Per Pfizer protocol</div>
    </div>
    """, unsafe_allow_html=True)
with t2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Experimental Median rPFS</div>
        <div class="metric-value">{exp_median:.2f} mo</div>
        <div style="color:#8b92a8; font-size:12px;">Implied by your HR</div>
    </div>
    """, unsafe_allow_html=True)
with t3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">rPFS Delta</div>
        <div class="metric-value" style="color:#2ecc71;">+{delta:.2f} mo</div>
        <div style="color:#8b92a8; font-size:12px;">Phase 1b delta: +8.1 mo</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SUMMARY VERDICT
# ============================================================
st.markdown("---")
st.markdown("## Verdict")

verdict_color = "#2ecc71" if pos > 0.65 else "#f39c12" if pos > 0.45 else "#e74c3c"
verdict_text = ("Strong Win Likely" if pos > 0.75 else
                "Probable Win" if pos > 0.60 else
                "Coin Flip" if pos > 0.45 else "Likely Loss")

st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%);
            padding:30px; border-radius:12px; border-left:6px solid {verdict_color};">
    <div style="color:{verdict_color}; font-size:14px; text-transform:uppercase; letter-spacing:2px; font-weight:700;">
        {verdict_text}
    </div>
    <div style="color:white; font-size:36px; font-weight:700; margin:10px 0;">
        {pos*100:.1f}% Probability of Success
    </div>
    <div style="color:#e8eaed; font-size:15px; line-height:1.6;">
        Central HR estimate: <span style="color:white; font-weight:700;">{adj_hr:.2f}</span> (vs Pfizer's design target of 0.66)<br>
        Implied median rPFS: <span style="color:white; font-weight:700;">{control_median:.1f} mo</span> (control) vs
        <span style="color:white; font-weight:700;">{exp_median:.1f} mo</span> (combo) — Δ {delta:.1f} mo<br>
        P(true HR &lt; 0.75): <span style="color:white; font-weight:700;">{p_below_075:.1f}%</span><br><br>
        <span style="color:#8b92a8;">
        Pfizer designed a trial with a wide runway after meaningful attenuation from the Phase 1b signal.
        The central question is whether the true HR stays below ~0.75. If correlated downside risks materialize
        (tougher control, more site dilution, greater regression), the trial slides into the borderline-to-failure zone.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    f"Built for the BAM/Longaeva Pfizer MEVPRO-1 case study. {len(wins):,}-draw Monte Carlo simulation. "
    "Power calculated using log-rank approximation. All inputs adjustable in the sidebar."
)
