import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt
import random

from utils.utils import (
    load_technology_data,
    load_price_data,
    load_net_capacity_data,
    load_co2_intensity,
    HOURS_PER_YEAR,
    TECH_COLORS,
)

seed = 42

np.random.seed(seed)

random.seed(seed)

# Instruct Gurobi to use the same seed for any randomized routines (best-effort;
# model may be created later, so use gp.setParam which applies globally)
try:
    gp.setParam("Seed", seed)
except Exception:
    pass

print(f"Reproducibility seed set to {seed}")

# ================================================================
#                  MAIN ENTRYPOINT FOR STREAMLIT
# ================================================================
def run_model_3(params):
    """
    Runs Model 3: Scenario & Robustness Analysis.

    Parameters expected (from Streamlit):
        - initial_budget : float (MEUR)
        - market_share : float (0–1)
        - demand_growth_rate : float
        - price_uncertainty : "Low" / "Medium" / "High"
        - num_scenarios : int
        - num_periods : int

    Returns:
        dict with:
            - status ("optimal", "not_optimal", or "error: ...")
            - kpis
            - tables
            - plots
    """

    # -----------------------------
    # Load core data
    # -----------------------------
    tech_units, tech_data = load_technology_data()
    net_cap_df, sum_capacity_mw = load_net_capacity_data()
    price_df = load_price_data(params["num_periods"])
    baseline_prices = price_df["Price"].values  # EUR/MWh

    co2_intensity = load_co2_intensity(tech_data)

    tech_names = list(tech_data.keys())
    print("-----------------------------")
    print("Loaded technologies:")
    print("Tech list:", tech_names)
    num_periods = params["num_periods"]
    time_periods = list(range(num_periods))

    num_scenarios = params["num_scenarios"]
    scenarios = list(range(num_scenarios))

    # -----------------------------
    # Price uncertainty level
    # -----------------------------
    level = params["price_uncertainty"]
    if level == "Low":
        sigma_prices = 0.05
    elif level == "High":
        sigma_prices = 0.25
    else:  # "Medium"
        sigma_prices = 0.15

    # Capture factor means (as in notebook)
    gamma_mean = {
        "Utility-scale PV":      0.85,
        "Onshore wind":          0.90,
        "Offshore wind (fixed)": 0.92,
        "OCGT - Natural gas":    1.05,
        "Diesel engine farm":    1.00,
        "Coal power plant":      1.00,
        "Nuclear power plant":   1.00,
    }

    # -----------------------------
    # Scenario generation
    # -----------------------------
    hub_price_scenarios = _generate_hub_price_scenarios(
        num_scenarios=num_scenarios,
        baseline_prices=baseline_prices,
        sigma=sigma_prices,
    )

    gamma = _generate_capture_factor_scenarios(
        tech_names=tech_names,
        gamma_mean=gamma_mean,
        num_scenarios=num_scenarios,
        num_periods=num_periods,
        gamma_sigma=0.05,
        rho_gamma=0.7,
    )

    # -----------------------------
    # Build and solve Gurobi model
    # -----------------------------
    try:
        model, model_vars = _build_model(
            tech_data=tech_data,
            co2_intensity=co2_intensity,
            tech_names=tech_names,
            time_periods=time_periods,
            scenarios=scenarios,
            params=params,
            hub_price_scenarios=hub_price_scenarios,
            gamma=gamma,
            sum_capacity_mw=sum_capacity_mw,
        )
        model.optimize()
    except Exception as e:
        return {"status": f"error: {e}"}

    if model.status != gp.GRB.OPTIMAL:
        return {"status": "not_optimal"}

    # -----------------------------
    # Extract results and build outputs
    # -----------------------------
    results = _extract_results(
        model=model,
        v=model_vars,
        tech_names=tech_names,
        time_periods=time_periods,
        scenarios=scenarios,
        price_baseline=baseline_prices,
        hub_price_scenarios=hub_price_scenarios,
    )

    return results


# ================================================================
#              PRICE & CAPTURE FACTOR SCENARIOS
# ================================================================
def _generate_hub_price_scenarios(num_scenarios, baseline_prices, sigma=0.15):
    """
    AR(1)-like stochastic yearly price paths around a baseline.
    baseline_prices: array of EUR/MWh
    returns: dict {s: [price_t]_t} in EUR/MWh
    """
    T = len(baseline_prices)
    scenarios = {}

    for s in range(num_scenarios):
        shock = 0.0
        price_path = []

        for t in range(T):
            eps = np.random.normal(0, sigma)
            shock = 0.7 * shock + eps
            price = baseline_prices[t] * (1 + shock)
            price = max(-20, min(price, 300))  # clamp

            price_path.append(price)

        scenarios[s] = price_path

    return scenarios


def _generate_capture_factor_scenarios(
    tech_names,
    gamma_mean,
    num_scenarios,
    num_periods,
    gamma_sigma=0.05,
    rho_gamma=0.7,
):
    """
    Generate tech-specific capture factor scenarios γ_{tech,s,t}.
    Returns dict gamma[(tech, s, t)].
    """
    gamma = {}

    for tech in tech_names:
        for s in range(num_scenarios):
            shock = 0.0
            for t in range(num_periods):
                eps = np.random.normal(0, gamma_sigma)
                shock = rho_gamma * shock + eps

                g = gamma_mean.get(tech, 1.0) * (1 + shock)
                g = max(0.5, min(1.3, g))  # keep realistic bounds

                gamma[(tech, s, t)] = g

    return gamma


# ================================================================
#                     BUILD GUROBI MODEL
# ================================================================
def _build_model(
    tech_data,
    co2_intensity,
    tech_names,
    time_periods,
    scenarios,
    params,
    hub_price_scenarios,
    gamma,
    sum_capacity_mw,
):

    num_periods = len(time_periods)
    hours_per_year = HOURS_PER_YEAR

    # Parameters from UI
    initial_budget = params["initial_budget"]     # MEUR
    market_share = params["market_share"]         # fraction
    demand_growth_rate = params["demand_growth_rate"]

    # Fixed assumptions as in notebook
    base_demand = market_share * 35_250_000       # MWh
    carbon_price = 83.0                           # EUR/tCO2
    carbon_increase = 5.0                         # EUR/tCO2 per year
    net_revenue_factor = 0.7

    # Tech-economic parameters
    capex = {
        tech: tech_data[tech]["nominal_investment_total"]    # MEUR/MW
        for tech in tech_names
    }
    elec_eff = {
        tech: tech_data[tech]["elec_eff"]
        for tech in tech_names
    }
    vom_meur = {
        tech: tech_data[tech]["variable_om_total"] / 1e6     # MEUR/MWh
        for tech in tech_names
    }
    fixed_om_meur = {
        tech: tech_data[tech]["fixed_om_total"] / 1e6        # MEUR/MW
        for tech in tech_names
    }

    # Demand per year
    demand = {
        t: base_demand * ((1 + demand_growth_rate) ** t)
        for t in time_periods
    }

    # Carbon price per year
    carbon_price_eur_per_tco2 = {
        t: carbon_price + carbon_increase * t
        for t in time_periods
    }
    carbon_price_meur_per_tco2 = {
        t: carbon_price_eur_per_tco2[t] / 1e6
        for t in time_periods
    }

    # Electricity prices: convert to MEUR/MWh
    price_scenarios_meur = {
        s: [p / 1e6 for p in hub_price_scenarios[s]]
        for s in scenarios
    }

    # Blocks
    blocks = [1, 2, 3]

    pi_block = {}
    for s in scenarios:
        for t in time_periods:
            p = price_scenarios_meur[s][t]
            pi_block[(s, t, 1)] = 1.00 * p
            pi_block[(s, t, 2)] = 0.85 * p
            pi_block[(s, t, 3)] = 0.60 * p

    # Block quantity limits per year (deterministic)
    Q_block = {}
    for t in time_periods:
        Q_block[(t, 1)] = 0.25 * demand[t]  # high-value block
        Q_block[(t, 2)] = 0.35 * demand[t]
        Q_block[(t, 3)] = 0.40 * demand[t]

    # Max capacity per tech (MW), as in notebook:
    # market_share * total system capacity * 0.2
    max_capacity = {
        "Diesel engine farm":         market_share*sum_capacity_mw*0.2,
        "OCGT - Natural gas":         market_share*sum_capacity_mw*0.2,
        "Coal power plant":           market_share*sum_capacity_mw*0.2,
        "Nuclear power plant":        market_share*sum_capacity_mw*0.2,
        "Onshore wind":               market_share*sum_capacity_mw*0.2,
        "Offshore wind (fixed)":      market_share*sum_capacity_mw*0.2,
        "Utility-scale PV":           market_share*sum_capacity_mw*0.2,
    }

    # -----------------------------
    # Build model
    # -----------------------------
    m = gp.Model("Model3_Scenario_Robustness")

    # Variables
    investment = m.addVars(tech_names, time_periods, name="Investment")
    capacity = m.addVars(tech_names, time_periods, name="Capacity")

    energy = m.addVars(tech_names, time_periods, scenarios, name="Energy")
    sales = m.addVars(tech_names, time_periods, blocks, scenarios, name="Sales")

    budget = m.addVars(time_periods, scenarios, name="Budget")
    revenue = m.addVars(time_periods, scenarios, name="Revenue")
    op_cost = m.addVars(time_periods, scenarios, name="OpCost")
    co2_cost = m.addVars(time_periods, scenarios, name="CO2Cost")

    fixed_om_cost = m.addVars(time_periods, name="FixedOMCost")
    capex_cost = m.addVars(time_periods, name="CapexCost")

    # -----------------------------
    # Capacity accumulation & limits
    # -----------------------------
    for tech in tech_names:
        # Initial capacity = investment at t=0
        m.addConstr(capacity[tech, 0] == investment[tech, 0])
        for t in time_periods[1:]:
            m.addConstr(
                capacity[tech, t]
                == capacity[tech, t - 1] + investment[tech, t]
            )

        for t in time_periods:
            m.addConstr(capacity[tech, t] <= max_capacity[tech])

    # -----------------------------
    # Production limits
    # -----------------------------
    for tech in tech_names:
        for t in time_periods:
            for s in scenarios:
                m.addConstr(
                    energy[tech, t, s]
                    <= capacity[tech, t] * elec_eff[tech] * hours_per_year
                )

    # -----------------------------
    # Cost definitions
    # -----------------------------
    for t in time_periods:

        # CAPEX (scenario-independent)
        m.addConstr(
            capex_cost[t]
            == gp.quicksum(investment[tech, t] * capex[tech] for tech in tech_names)
        )

        # Fixed O&M (scenario-independent)
        m.addConstr(
            fixed_om_cost[t]
            == gp.quicksum(capacity[tech, t] * fixed_om_meur[tech] for tech in tech_names)
        )

        for s in scenarios:
            # Variable O&M
            m.addConstr(
                op_cost[t, s]
                == gp.quicksum(energy[tech, t, s] * vom_meur[tech] for tech in tech_names)
            )

            # CO2 cost
            m.addConstr(
                co2_cost[t, s]
                == gp.quicksum(
                    energy[tech, t, s]
                    * co2_intensity[tech]
                    * carbon_price_meur_per_tco2[t]
                    for tech in tech_names
                )
            )

    # -----------------------------
    # Sales = energy allocation to blocks
    # -----------------------------
    for tech in tech_names:
        for t in time_periods:
            for s in scenarios:
                m.addConstr(
                    gp.quicksum(sales[tech, t, b, s] for b in blocks)
                    == energy[tech, t, s]
                )

    # -----------------------------
    # Block capacity limits
    # -----------------------------
    for t in time_periods:
        for b in blocks:
            for s in scenarios:
                m.addConstr(
                    gp.quicksum(sales[tech, t, b, s] for tech in tech_names)
                    <= Q_block[(t, b)]
                )

    # -----------------------------
    # Revenue with capture factors
    # -----------------------------
    for t in time_periods:
        for s in scenarios:
            m.addConstr(
                revenue[t, s]
                == gp.quicksum(
                    sales[tech, t, b, s]
                    * pi_block[(s, t, b)]
                    * gamma[(tech, s, t)]
                    * net_revenue_factor
                    for tech in tech_names
                    for b in blocks
                )
            )

    # -----------------------------
    # Budget recursion (scenario-dependent)
    # -----------------------------
    for s in scenarios:
        # Initial budget
        m.addConstr(budget[0, s] == initial_budget)

        for t in time_periods:
            # Investment cannot exceed budget of scenario s at time t
            m.addConstr(capex_cost[t] <= budget[t, s])
            # Budget non-negativity
            m.addConstr(budget[t, s] >= 0)

            if t < num_periods - 1:
                m.addConstr(
                    budget[t + 1, s]
                    == budget[t, s]
                    - capex_cost[t]
                    + revenue[t, s]
                    - op_cost[t, s]
                    - fixed_om_cost[t]
                    - co2_cost[t, s]
                )

    # -----------------------------
    # Objective: Expected Profit
    # -----------------------------
    S = len(scenarios)
    m.setObjective(
        (1 / S)
        * gp.quicksum(
            revenue[t, s]
            - op_cost[t, s]
            - co2_cost[t, s]
            - fixed_om_cost[t]
            - capex_cost[t]
            for t in time_periods
            for s in scenarios
        ),
        gp.GRB.MAXIMIZE,
    )

    return m, {
        "investment": investment,
        "capacity": capacity,
        "energy": energy,
        "sales": sales,
        "budget": budget,
        "revenue": revenue,
        "op_cost": op_cost,
        "co2_cost": co2_cost,
        "fixed_om_cost": fixed_om_cost,
        "capex_cost": capex_cost,
        "time_periods": time_periods,
        "scenarios": scenarios,
        "tech_names": tech_names,
    }


# ================================================================
#                        EXTRACT RESULTS
# ================================================================
def _extract_results(model, v, tech_names, time_periods, scenarios, price_baseline, hub_price_scenarios):

    investment = v["investment"]
    capacity = v["capacity"]
    energy = v["energy"]
    revenue = v["revenue"]
    op_cost = v["op_cost"]
    co2_cost = v["co2_cost"]
    fixed_om_cost = v["fixed_om_cost"]
    capex_cost = v["capex_cost"]
    budget = v["budget"]

    # --------------------------------------
    # KPIs: profit distribution, CVaR, etc.
    # --------------------------------------
    profit = []
    for s in scenarios:
        total = sum(
            revenue[t, s].X
            - op_cost[t, s].X
            - co2_cost[t, s].X
            - fixed_om_cost[t].X
            - capex_cost[t].X
            for t in time_periods
        )
        profit.append(total)

    profit = np.array(profit)
    expected_profit = float(profit.mean())
    # CVaR 10%: mean of worst 10% scenarios
    alpha = 0.10
    n_cvar = max(1, int(alpha * len(profit)))
    cvar10 = float(np.sort(profit)[:n_cvar].mean())

    final_budget = float(
        np.mean([budget[time_periods[-1], s].X for s in scenarios])
    )

    # Expected mean revenue over all years & scenarios
    # expected_revenue = float(
    #    np.mean([revenue[t, s].X for t in time_periods for s in scenarios])
    #)
    # Expected TOTAL revenue over full horizon (MEUR)
    expected_revenue = float(
        np.mean([
            sum(revenue[t, s].X for t in time_periods)
            for s in scenarios
        ])
    )

    # Final-year installed capacity (MW)
    cap_final = {tech: capacity[tech, time_periods[-1]].X for tech in tech_names}
    total_capacity = float(sum(cap_final.values()))

    # --------------------------------------
    # TABLES
    # --------------------------------------

    # 1) Installed capacity over time (MW)
    cap_table = pd.DataFrame(
        {
            f"Year {t}": {tech: capacity[tech, t].X for tech in tech_names}
            for t in time_periods
        }
    )
    cap_table.index.name = "Technology"
    cap_table.loc["Total"] = cap_table.sum(axis=0)
    cap_table = cap_table.round(1)

    # 2) Expected production by tech & year (GWh)
    prod_data = {}
    for tech in tech_names:
        prod_data[tech] = [
            np.mean([energy[tech, t, s].X for s in scenarios]) / 1e3
            for t in time_periods
        ]

    prod_table = pd.DataFrame(prod_data, index=time_periods).T
    prod_table.columns = [f"Year {t}" for t in time_periods]
    prod_table.loc["Total"] = prod_table.sum(axis=0)
    prod_table = prod_table.round(1)

    # 3) Profit distribution by scenario (MEUR)
    profit_table = pd.DataFrame(
        {
            "Scenario": scenarios,
            "Profit (MEUR)": profit,
        }
    ).set_index("Scenario")
    profit_table = profit_table.round(1)

    # 4) Economic summary per year (MEUR)
    economic = pd.DataFrame(
        {
            "Revenue": [
                np.mean([revenue[t, s].X for s in scenarios])
                for t in time_periods
            ],
            "Variable OPEX": [
                np.mean([op_cost[t, s].X for s in scenarios])
                for t in time_periods
            ],
            "Fixed O&M": [fixed_om_cost[t].X for t in time_periods],
            "CAPEX": [capex_cost[t].X for t in time_periods],
            "CO2 Cost": [
                np.mean([co2_cost[t, s].X for s in scenarios])
                for t in time_periods
            ],
        }
    )
    economic["Net Cashflow"] = (
        economic["Revenue"]
        - economic["Variable OPEX"]
        - economic["Fixed O&M"]
        - economic["CAPEX"]
        - economic["CO2 Cost"]
    )
    economic.index = [f"Year {t}" for t in time_periods]
    economic = economic.round(1)

    # --------------------------------------
    # PLOTS
    # --------------------------------------
    plots = {}

    # === Capacity mix pie chart (final year) ===
    # Filter out technologies with zero capacity
    cap_filtered = {tech: val for tech, val in cap_final.items() if val > 1e-6}

    fig1, ax = plt.subplots(figsize=(6, 6))

    labels = list(cap_filtered.keys())
    sizes = list(cap_filtered.values())

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=[TECH_COLORS.get(tech, "#CCCCCC") for tech in labels],
    )
    ax.set_title("Installed Capacity Mix (Final Year)")

    plots["capacity_pie"] = fig1
    # === Expected Generation per Technology (GWh) ===
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    for tech in tech_names:
        # Expected production (scenario average)
        mean_prod = np.array([
            np.mean([energy[tech, t, s].X for s in scenarios])
            for t in time_periods
        ]) / 1e3  # convert MWh → GWh

        ax4.plot(
            time_periods,
            mean_prod,
            linewidth=2,
            label=tech,
            color=TECH_COLORS.get(tech, "#CCCCCC")
        )

    ax4.set_title("Expected Annual Generation per Technology")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Generation (GWh)")
    ax4.set_xticks(time_periods)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.225))

    plots["expected_generation_per_tech"] = fig4

    # === Financial flow plot (Revenue positive, costs negative, net line) ===
    rev_vals = economic["Revenue"].values
    op_vals = economic["Variable OPEX"].values
    fix_vals = economic["Fixed O&M"].values
    capx_vals = economic["CAPEX"].values
    co2_vals = economic["CO2 Cost"].values
    net_vals = economic["Net Cashflow"].values

    x = np.arange(len(time_periods))

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # Revenue (positive)
    ax3.bar(x, rev_vals, label="Revenue", color="tab:green")

    # Costs (negative, stacked downward)
    ax3.bar(x, -op_vals, label="Variable OPEX", color="tab:orange")
    ax3.bar(x, -fix_vals, bottom=-op_vals, label="Fixed O&M", color="tab:gray")
    ax3.bar(
        x,
        -capx_vals,
        bottom=-op_vals - fix_vals,
        label="CAPEX",
        color="tab:blue",
    )
    ax3.bar(
        x,
        -co2_vals,
        bottom=-op_vals - fix_vals - capx_vals,
        label="CO₂ Cost",
        color="tab:red",
    )

    # Net cashflow line
    ax3.plot(x, net_vals, marker="o", color="black", label="Net Cashflow")

    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Financial Flows per Year")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("MEUR")
    ax3.set_xticks(x)
    ax3.set_xticklabels(time_periods)
    ax3.grid(True, axis="y")
    ax3.legend(loc="lower center", ncol=4)
    plots["financial_flows"] = fig3

    # === PRICE UNCERTAINTY PLOT ===
    price_matrix = np.array([
        hub_price_scenarios[s] for s in scenarios
    ])  # shape: (S, T)

    mean_price = price_matrix.mean(axis=0)
    min_price  = price_matrix.min(axis=0)
    max_price  = price_matrix.max(axis=0)
    p10 = np.percentile(price_matrix, 10, axis=0)
    p90 = np.percentile(price_matrix, 90, axis=0)

    fig_price, ax = plt.subplots(figsize=(10, 5))

    # Baseline path
    ax.plot(time_periods, price_baseline, linestyle="--",
            label="Baseline Price", color="black")

    # Mean scenario price
    ax.plot(time_periods, mean_price, label="Scenario Mean", color="tab:blue")

    # Min–max shading
    ax.fill_between(time_periods, min_price, max_price,
                    alpha=0.2, label="Min–Max Band", color="tab:blue")

    # 10–90% band
    ax.fill_between(time_periods, p10, p90,
                    alpha=0.3, label="10–90% Band", color="tab:orange")

    ax.set_xlabel("Year")
    ax.set_ylabel("Hub Price (EUR/MWh)")
    ax.set_title("Stochastic Wholesale Price Scenarios (Yearly)")
    ax.grid(True)
    ax.legend()

    plots["price_uncertainty"] = fig_price

    # --------------------------------------
    # Return full result dict
    # --------------------------------------
    return {
        "status": "optimal",
        "kpis": {
            "expected_profit": expected_profit,
            "cvar10": cvar10,
            "final_budget": final_budget,
            "expected_revenue": expected_revenue,
            "total_capacity": total_capacity,
        },
        "tables": {
            "capacity": cap_table,
            "production": prod_table,
            "profit_distribution": profit_table,
            "economic": economic,
        },
        "plots": plots,
    }