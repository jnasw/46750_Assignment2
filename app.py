import streamlit as st

from utils.model_1 import run_model_1
from utils.model_2 import run_model_2
from utils.model_3 import run_model_3

import numpy as np
import random
import gurobipy as gp

# GLOBAL REPRODUCIBILITY
np.random.seed(42)
random.seed(42)
gp.setParam("Seed", 42)


st.set_page_config(
    page_title="Optimisation Models Dashboard",
    layout="wide"
)

st.title("⚡ Portfolio Optimisation Dashboard")

# ---------------------------------------------------------
# Tabs for the three models
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Model 1", "Model 2", "Model 3"])


# =========================================================
#                       MODEL 1 TAB
# =========================================================
with tab1:

    st.header("Model 1 — Cost-Based Optimal Technology Mix")

    st.subheader("Parameters")

    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input(
            "Investment Budget (MEUR)",
            min_value=0.0,
            max_value=2000.0,
            value=500.0,
            step=10.0,
        )

        total_demand = st.number_input(
            "Total Energy Demand (MWh)",
            min_value=0.0,
            max_value=20_000_000.0,
            value=1_000_000.0,
            step=100_000.0,
        )

    with col2:
        min_num_technologies = st.number_input(
            "Minimum Number of Technologies",
            min_value=1,
            max_value=10,
            value=4,
        )

        min_share = st.slider(
            "Minimum Share per Technology",
            min_value=0.0,
            max_value=0.50,
            value=0.10,
        )

    params1 = {
        "budget": budget,
        "total_demand": total_demand,
        "min_num_technologies": min_num_technologies,
        "min_share_per_tech": min_share,
    }

    st.markdown("---")

    if st.button("Run Model 1"):
        st.subheader("Results")

        results1 = run_model_1(params1)

        if results1["status"] != "optimal":
            st.error("Model 1 did not solve to optimality.")
        else:
            st.success("Model 1 — optimal solution found!")

            st.metric(
                "Total Annual Cost",
                f"{results1['objective_value']:.2f} MEUR",
            )

            st.subheader("Technology Results")
            st.dataframe(
                results1["tables"]["results"],
                use_container_width=True,
            )

            st.subheader("Visualizations")
            c1, c2 = st.columns([1, 2])

            with c1:
                st.markdown("**Energy Distribution by Technology**")
                st.pyplot(results1["plots"]["energy_distribution"])

            with c2:
                st.markdown("**Investment vs Energy Produced**")
                st.pyplot(results1["plots"]["investment_vs_energy"])


# =========================================================
#                       MODEL 2 TAB
# =========================================================
with tab2:

    st.header("Model 2 — Dynamic Investment and Revenue Maximization")

    st.subheader("Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_budget = st.number_input(
            "Initial Budget (MEUR)",
            min_value=0.0,
            max_value=5000.0,
            value=100.0,
            step=10.0,
        )

        market_share = st.slider(
            "Market Share",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )

    with col2:
        demand_growth_rate = st.slider(
            "Demand Growth Rate (per year)",
            min_value=0.0,
            max_value=0.10,
            value=0.02,
            step=0.005,
        )

        num_periods = st.slider(
            "Time Horizon (years)",
            min_value=5,
            max_value=35,
            value=20,
        )

    with col3:
        max_capacity_share = st.slider(
            "Max Capacity Share (per tech)",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
        )

        net_revenue_factor = st.slider(
            "Net Revenue Factor",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )

    params2 = {
        "initial_budget": initial_budget,
        "market_share": market_share,
        "demand_growth_rate": demand_growth_rate,
        "num_periods": num_periods,
        "max_capacity_share": max_capacity_share,
        "net_revenue_factor": net_revenue_factor,
    }

    st.markdown("---")

    if st.button("Run Model 2"):
        st.subheader("Results")

        results2 = run_model_2(params2)

        if results2["status"] != "optimal":
            st.error("Model 2 did not solve to optimality.")
        else:
            st.success("Model 2 — optimal solution found!")

            # KPIs
            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Total Profit",
                f"{results2['kpis']['total_profit']:.2f} MEUR",
            )
            k2.metric(
                "Final Budget",
                f"{results2['kpis']['final_budget']:.2f} MEUR",
            )
            k3.metric(
                "Total Revenue",
                f"{results2['kpis']['total_revenue']:.2f} MEUR",
            )

    
            # Create side-by-side columns (50/50)
            col1, col2 = st.columns([1, 1])

            # --- LEFT PLOT: Installed Capacity ---
            with col1:
                st.markdown("Installed Capacity Over Time")
                st.pyplot(results2["plots"]["capacity_over_time"])

            # --- RIGHT PLOT: Financial Flow ---
            with col2:
                st.markdown("Financial Flow per Period")
                st.pyplot(results2["plots"]["financial_flows"])

            # Tables in expanders
            st.subheader("Detailed Tables")

            with st.expander("Installed Capacity (MW)"):
                st.dataframe(
                    results2["tables"]["capacity"],
                    use_container_width=True,
                )

            with st.expander("Energy Production (GWh)"):
                st.dataframe(
                    results2["tables"]["production"],
                    use_container_width=True,
                )

            with st.expander("Financial Flows per Period (MEUR)"):
                st.dataframe(
                    results2["tables"]["financials"],
                    use_container_width=True,
                )


# =========================================================
#                       MODEL 3 TAB
# =========================================================
with tab3:

    st.header("Model 3 — Stochastic Revenue Optimization")

    st.subheader("Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_budget = st.number_input(
            "Initial Budget (MEUR)",
            min_value=0.0,
            max_value=5000.0,
            value=150.0,
            key="m3_initial_budget",
            step=10.0,
        )
        market_share = st.slider(
            "Market Share",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            key="m3_market_share",
        )

    with col2:
        demand_growth_rate = st.slider(
            "Demand Growth Rate",
            min_value=0.0,
            max_value=0.10,
            value=0.02,
            step=0.005,
            key="m3_demand_growth",
        )
        uncertainty = st.selectbox(
            "Price Uncertainty Level",
            ["Low", "Medium", "High"],
            index=1,
            key="m3_uncertainty",
        )

    with col3:
        num_scenarios = st.slider(
            "Number of Scenarios",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            key="m3_num_scenarios",
        )
        num_periods = st.slider(
            "Time Horizon (years)",
            min_value=5,
            max_value=35,
            value=20,
            key="m3_num_periods",
        )

    params3 = {
        "initial_budget": initial_budget,
        "market_share": market_share,
        "demand_growth_rate": demand_growth_rate,
        "price_uncertainty": uncertainty,
        "num_scenarios": num_scenarios,
        "num_periods": num_periods,
    }

    st.markdown("---")

    if st.button("Run Model 3"):
        st.subheader("Results")

        results3 = run_model_3(params3)

        if results3["status"] != "optimal":
            st.error("Model 3 did not solve to optimality.")
        else:
            st.success("Model 3 — optimal solution found!")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Expected Profit", f"{results3['kpis']['expected_profit']:.2f} MEUR")
            k2.metric("CVaR (10%)", f"{results3['kpis']['cvar10']:.2f} MEUR", )
            k3.metric("Final Budget", f"{results3['kpis']['final_budget']:.2f} MEUR")
            k4.metric("Expected Revenue", f"{results3['kpis']['expected_revenue']:.2f} MEUR")
            k5.metric("Total Capacity", f"{results3['kpis']['total_capacity']:.2f} MW")

            colA, colB = st.columns([1, 2])

            with colA:
                st.markdown("Installed Capacity Mix")
                st.pyplot(results3["plots"]["capacity_pie"])

            with colB:
                st.markdown("Expected Generation")
                st.pyplot(results3["plots"]["expected_generation_per_tech"])

            colA, colB = st.columns([1, 1])

            with colA:
                st.markdown("Uncertainty Analysis")
                st.pyplot(results3["plots"]["price_uncertainty"])

            with colB:
                st.markdown("Financial Flow per Period")
                st.pyplot(results3["plots"]["financial_flows"])

            st.subheader("Detailed Tables")
            with st.expander("Installed Capacity"):
                st.dataframe(results3["tables"]["capacity"], use_container_width=True)

            with st.expander("Expected Production"):
                st.dataframe(results3["tables"]["production"], use_container_width=True)

            #with st.expander("Profit Distribution"):
            #    st.dataframe(results3["tables"]["profit_distribution"], use_container_width=True)
            
            with st.expander("Economic Summary per Year (MEUR)"):
                st.dataframe(results3["tables"]["economic"], use_container_width=True)