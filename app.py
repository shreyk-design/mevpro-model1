import streamlit as st
import plotly.graph_objects as go
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="MEVPRO-1 Case Runner", layout="wide")

st.title("Pfizer MEVPRO-1 (mCRPC) Phase 3 Case Runner")
st.markdown("### Sensitivity Analysis & HR Walk-Down Dashboard")
st.markdown("Adjust the clinical variables in the sidebar to stress-test the Phase 1b data and estimate Phase 3 Statistical Power (Target: 302 events).")

# 2. Sidebar Controls
st.sidebar.header("Phase 3 HR Adjustments")
st.sidebar.markdown("Define the penalties applied to the Phase 1b baseline.")

base_hr = 0.51

ecog = st.sidebar.slider("ECOG Imbalance Normalization", 0.0, 0.10, 0.04, 0.01)
reg_mean = st.sidebar.slider("Small-N Regression to Mean", 0.0, 0.15, 0.09, 0.01)
docetaxel = st.sidebar.slider("Docetaxel Control Arm Impact", 0.0, 0.10, 0.03, 0.01)
site_dilution = st.sidebar.slider("Community Site Dilution", 0.0, 0.05, 0.02, 0.01)
bicr = st.sidebar.slider("BICR Strictness Penalty", 0.0, 0.05, 0.02, 0.01)

# 3. Math & Logic
total_penalty = ecog + reg_mean + docetaxel + site_dilution + bicr
adj_hr = base_hr + total_penalty

# Statistical Power Interpolation (Based on MEVPRO-1 302 Events)
hr_points = [0.50, 0.60, 0.66, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
power_points = [0.99, 0.98, 0.95, 0.87, 0.70, 0.49, 0.29, 0.15, 0.05]
power_est = np.interp(adj_hr, hr_points, power_points)

# Determine the qualitative outcome
if power_est >= 0.85:
    status_color = "🟢"
    status_text = "High Confidence Win"
elif power_est >= 0.70:
    status_color = "🟡"
    status_text = "Probable Win (Standard Powering)"
elif power_est >= 0.50:
    status_color = "🟠"
    status_text = "Toss-Up / Borderline"
else:
    status_color = "🔴"
    status_text = "Underpowered (Likely Trial Failure)"

# 4. Top Metric Row
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Base Phase 1b HR", f"{base_hr:.2f}")
col2.metric("Cumulative Clinical Penalty", f"+{total_penalty:.2f}")
col3.metric("Adjusted Phase 3 HR", f"{adj_hr:.2f}")

# 5. Power Readout
st.markdown("---")
st.subheader(f"Estimated Trial Power: {power_est*100:.1f}%")
st.markdown(f"**Outcome Projection:** {status_color} {status_text}")

# 6. Waterfall Chart
st.markdown("### The HR Walk-Down")
fig = go.Figure(go.Waterfall(
    name="MEVPRO-1 HR", 
    orientation="v",
    measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
    x=["Phase 1b Baseline", "ECOG Normalization", "Regression to Mean", "Docetaxel Squeeze", "Site Dilution", "BICR Strictness", "Target Phase 3 HR"],
    text=[f"{base_hr:.2f}", f"+{ecog:.2f}", f"+{reg_mean:.2f}", f"+{docetaxel:.2f}", f"+{site_dilution:.2f}", f"+{bicr:.2f}", f"{adj_hr:.2f}"],
    textposition="outside",
    y=[base_hr, ecog, reg_mean, docetaxel, site_dilution, bicr, adj_hr],
    decreasing={"marker":{"color":"#2ca02c"}},
    increasing={"marker":{"color":"#d62728"}},
    totals={"marker":{"color":"#1f77b4"}},
    connector={"line":{"color":"rgb(63, 63, 63)"}}
))

fig.update_layout(
    title="Clinical Penalty Waterfall",
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(title="Hazard Ratio", range=[0.4, 1.0])
)

st.plotly_chart(fig, use_container_width=True)
