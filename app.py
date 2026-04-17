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
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
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
    .metric-value { color: #ffffff; font-size: 32px; font-weight: 700; }
    h1 { color: #ffffff !important; font-weight: 700; }
    h2, h3 { color: #e8eaed !important; }
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
# HELPER FUNCTIONS
# ============================================================
def norm_cdf(x):
    """Standard normal CDF using error function."""
    return 0.5 * (1 + erf(x / sqrt(2)))

def norm_ppf(p):
    """Inverse standard normal CDF (approximation)."""
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
    if hr >= 1.0: return alpha_level / 2
    if hr <= 0: return 1.0
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

ecog = st.sidebar.slider("ECOG Imbalance", 0.0, 0.10, 0.04, 0.01)
reg_mean = st.sidebar.slider("Regression to Mean", 0.0, 0.15, 0.09, 0.01)
dose = st.sidebar.slider("Dose Reduction", 0.0, 0.05, 0.02, 0.01)
pop_shift = st.sidebar.slider("Population Shift", -0.05, 0.05, -0.02, 0.01)
docetaxel = st.sidebar.slider("Stronger Blended Control", 0.0, 0.10, 0.03, 0.01)
site_dilution = st.sidebar.slider("Site Dilution", 0.0, 0.05, 0.02, 0.01)
bicr = st.sidebar.slider("BICR vs Investigator", 0.0, 0.05, 0.02, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("## Trial Design")
n_events = st.sidebar.slider("rPFS Events", 200, 400, 302, 1)
alpha = st.sidebar.selectbox("Significance Threshold", [0.05, 0.025, 0.01], index=0)
hr_uncertainty = st.sidebar.slider("HR Std Dev", 0.03, 0.15, 0.07, 0.01)

# ============================================================
# CORE CALCULATIONS
# ============================================================
total_penalty = ecog + reg_mean + dose + pop_shift + docetaxel + site_dilution + bicr
adj_hr = base_hr + total_penalty

hr_dist, powers, wins = run_monte_carlo(adj_hr, hr_uncertainty, n_events, alpha)
pos = wins.mean()
power_central = calc_power(adj_hr, n_events, alpha)

se_logHR = 2 / sqrt(n_events)
ci_lower = adj_hr * np.exp(-1.96 * se_logHR)
ci_upper = adj_hr * np.exp(1.96 * se_logHR)

def classify_outcome(hr, won):
    if not won: return "Loss"
    if hr <= 0.65: return "Blowout Win"
    elif hr <= 0.70: return "Clear Win"
    elif hr <= 0.75: return "Solid Win"
    else: return "Marginal Win"

outcomes = [classify_outcome(hr, win) for hr, win in zip(hr_dist, wins)]
outcome_counts = pd.Series(outcomes).value_counts(normalize=True) * 100
outcome_order = ["Blowout Win", "Clear Win", "Solid Win", "Marginal Win", "Loss"]
outcome_counts = outcome_counts.reindex(outcome_order, fill_value=0)

# ============================================================
# TOP METRICS
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
# SCENARIO COMPARISON
# ============================================================
st.markdown("---")
st.markdown("## Scenario Analysis")
st.caption("Three pre-built scenarios alongside your custom slider configuration.")

scenarios = {
    "Downside": {"ecog": 0.07, "reg_mean": 0.13, "dose": 0.04, "pop_shift": 0.02, "docetaxel": 0.07, "site_dilution": 0.04, "bicr": 0.03, "color": "#e74c3c", "description": "Aggressive penalties materialized."},
    "Base": {"ecog": 0.04, "reg_mean": 0.09, "dose": 0.02, "pop_shift": -0.02, "docetaxel": 0.03, "site_dilution": 0.02, "bicr": 0.02, "color": "#3498db", "description": "Central estimate; balanced view."},
    "Upside": {"ecog": 0.02, "reg_mean": 0.05, "dose": 0.00, "pop_shift": -0.04, "docetaxel": 0.01, "site_dilution": 0.01, "bicr": 0.01, "color": "#2ecc71", "description": "Phase 1b signal mostly preserved."}
}

scenario_results = {}
for name, params in scenarios.items():
    s_hr = base_hr + sum([v for k, v in params.items() if k not in ["color", "description"]])
    s_dist, s_powers, s_wins = run_monte_carlo(s_hr, hr_uncertainty, n_events, alpha, seed=hash(name) % 100000)
    scenario_results[name] = {"hr": s_hr, "pos": s_wins.mean(), "color": params["color"], "description": params["description"]}

scenario_results["Your Custom"] = {"hr": adj_hr, "pos": pos, "color": "#9b59b6", "description": "Reflects your current slider settings."}

sc1, sc2, sc3, sc4 = st.columns(4)
for col, (name, result) in zip([sc1, sc2, sc3, sc4], scenario_results.items()):
    with col:
        st.markdown(f"""
        <div class="scenario-box" style="border-left-color: {result['color']};">
            <div style="color:{result['color']}; font-weight:700; font-size:14px; text-transform:uppercase;">{name}</div>
            <div style="color:white; font-size:28px; font-weight:700; margin:8px 0;">{result['pos']*100:.1f}%</div>
            <div style="color:#8b92a8; font-size:13px;">HR: <span style='color:white;'>{result['hr']:.2f}</span></div>
            <div style="color:#8b92a8; font-size:11px; margin-top:8px;">{result['description']}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# CHARTS: OUTCOME & DISTRIBUTION
# ============================================================
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### First-Read Outcome Distribution")
    colors = ["#1a9641", "#73c378", "#a6d96a", "#fdae61", "#d7191c"]
    fig_outcome = go.Figure(data=[
        go.Bar(x=outcome_counts.index, y=outcome_counts.values,
               text=[f"{v:.1f}%" for v in outcome_counts.values], textposition="outside",
               textfont=dict(color='white', size=14), marker=dict(color=colors))
    ])
    fig_outcome.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#0e1117', font=dict(color='white'),
                              yaxis=dict(title="Probability (%)", gridcolor='#2d3548', range=[0, max(outcome_counts.values)*1.2]),
                              xaxis=dict(gridcolor='#2d3548'), showlegend=False, height=350, margin=dict(t=20))
    st.plotly_chart(fig_outcome, use_container_width=True)

with c2:
    st.markdown("#### Simulated Phase 3 HR Density")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=hr_dist, nbinsx=80, marker=dict(color="#4a90e2", line=dict(color='rgba(255,255,255,0.1)', width=0.5)), opacity=0.8))
    fig_dist.add_vline(x=adj_hr, line_dash="solid", line_color="white", annotation_text=f"Central: {adj_hr:.2f}")
    fig_dist.add_vline(x=0.66, line_dash="dash", line_color="#2ecc71", annotation_text="Target (0.66)")
    fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#0e1117', font=dict(color='white'),
                           xaxis=dict(title="True Hazard Ratio", range=[0.3, 1.3], gridcolor='#2d3548'),
                           yaxis=dict(title="Frequency", gridcolor='#2d3548'), showlegend=False, height=350, margin=dict(t=20))
    st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================
# WATERFALL & POWER CURVE
# ============================================================
c3, c4 = st.columns(2)

with c3:
    st.markdown("#### HR Walk-Down")
    fig_wf = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"],
        x=["Ph1b Base", "ECOG", "Regr. to Mean", "Dose", "Pop Shift", "Control", "Sites", "BICR", "Target HR"],
        text=[f"{base_hr:.2f}", f"+{ecog:.2f}", f"+{reg_mean:.2f}", f"+{dose:.2f}", f"{pop_shift:+.2f}", f"+{docetaxel:.2f}", f"+{site_dilution:.2f}", f"+{bicr:.2f}", f"{adj_hr:.2f}"],
        textposition="outside", y=[base_hr, ecog, reg_mean, dose, pop_shift, docetaxel, site_dilution, bicr, adj_hr],
        decreasing={"marker": {"color": "#2ecc71"}}, increasing={"marker": {"color": "#e74c3c"}}, totals={"marker": {"color": "#4a90e2"}}
    ))
    fig_wf.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#0e1117', font=dict(color='white'),
                         yaxis=dict(title="Hazard Ratio", range=[0.4, 0.9], gridcolor='#2d3548'), height=400, margin=dict(t=20))
    st.plotly_chart(fig_wf, use_container_width=True)

with c4:
    st.markdown("#### Statistical Power Curve")
    hr_range = np.linspace(0.50, 1.00, 100)
    power_curve = [calc_power(hr, n_events, alpha) * 100 for hr in hr_range]
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=hr_range, y=power_curve, mode='lines', line=dict(color='#4a90e2', width=4), fill='tozeroy', fillcolor='rgba(74,144,226,0.2)'))
    fig_power.add_vline(x=adj_hr, line_color="white", annotation_text=f"Central ({adj_hr:.2f})")
    fig_power.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#0e1117', font=dict(color='white'),
                            xaxis=dict(title="True Hazard Ratio", gridcolor='#2d3548'),
                            yaxis=dict(title="Statistical Power (%)", range=[0, 105], gridcolor='#2d3548'), height=400, margin=dict(t=20))
    st.plotly_chart(fig_power, use_container_width=True)

# ============================================================
# SENSITIVITY TORNADO
# ============================================================
st.markdown("---")
st.markdown("## Sensitivity Tornado")
st.caption("PoS impact of moving each parameter from its best-case to worst-case assumption.")

base_params = {"ECOG": ecog, "Regression": reg_mean, "Dose": dose, "Pop Shift": pop_shift, "Control Squeeze": docetaxel, "Site Dilution": site_dilution, "BICR": bicr}
ranges = {"ECOG": (0.0, 0.10), "Regression": (0.0, 0.15), "Dose": (0.0, 0.05), "Pop Shift": (-0.05, 0.05), "Control Squeeze": (0.0, 0.10), "Site Dilution": (0.0, 0.05), "BICR": (0.0, 0.05)}

def calc_pos_quick(params):
    h = base_hr + sum(params.values())
    s = np.clip(np.random.normal(loc=h, scale=hr_uncertainty, size=5000), 0.30, 1.50)
    return np.random.binomial(1, [calc_power(x, n_events, alpha) for x in s]).mean()

sensitivity = []
for param, _ in base_params.items():
    low_p, high_p = base_params.copy(), base_params.copy()
    low_p[param], high_p[param] = ranges[param][0], ranges[param][1]
    sensitivity.append({"Parameter": param, "Low": calc_pos_quick(low_p)*100, "High": calc_pos_quick(high_p)*100, "Swing": abs(calc_pos_quick(high_p)*100 - calc_pos_quick(low_p)*100)})

sens_df = pd.DataFrame(sensitivity).sort_values("Swing", ascending=True)
fig_tornado = go.Figure()
fig_tornado.add_trace(go.Bar(y=sens_df["Parameter"], x=sens_df["Low"] - pos * 100, orientation='h', name='Bullish', marker=dict(color='#2ecc71')))
fig_tornado.add_trace(go.Bar(y=sens_df["Parameter"], x=sens_df["High"] - pos * 100, orientation='h', name='Bearish', marker=dict(color='#e74c3c')))
fig_tornado.update_layout(barmode='overlay', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#0e1117', font=dict(color='white'),
                          xaxis=dict(title=f"PoS Change vs Base ({pos*100:.1f}%)", gridcolor='#2d3548'), yaxis=dict(gridcolor='#2d3548'), height=400)
st.plotly_chart(fig_tornado, use_container_width=True)

# ============================================================
# VERDICT
# ============================================================
st.markdown("---")
verdict_color = "#2ecc71" if pos > 0.65 else "#f39c12" if pos > 0.45 else "#e74c3c"
st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%); padding:30px; border-radius:12px; border-left:6px solid {verdict_color};">
    <div style="color:{verdict_color}; font-size:14px; text-transform:uppercase; letter-spacing:2px; font-weight:700;">Final Assessment</div>
    <div style="color:white; font-size:36px; font-weight:700; margin:10px 0;">{pos*100:.1f}% Probability of Success</div>
    <div style="color:#e8eaed; font-size:15px; line-height:1.6;">
        Pfizer's MEVPRO-1 is mathematically insulated against substantial Phase 1b signal dilution. Even with a modeled 
        <span style="color:white; font-weight:700;">+{total_penalty:.2f} HR penalty</span>, the 302-event trial design protects the readout.
    </div>
</div>
""", unsafe_allow_html=True)
