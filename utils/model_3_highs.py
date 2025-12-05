import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import value

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
print(f"Reproducibility seed set to {seed}")


# ============================================================================
#                              MAIN ENTRYPOINT
# ============================================================================
def run_model_3(params):

    # ---------- Load data ----------
    tech_units, tech_data = load_technology_data()
    net_cap_df, sum_capacity_mw = load_net_capacity_data()
    price_df = load_price_data(params["num_periods"])
    baseline_prices = price_df["Price"].values

    co2_intensity = load_co2_intensity(tech_data)
    # Fix NaN CO2 intensity values → treat as zero
    for tech, val in co2_intensity.items():
        if val != val:   # check for NaN
            co2_intensity[tech] = 0.0
    tech_names = list(tech_data.keys())

    num_periods = params["num_periods"]
    time_periods = list(range(num_periods))

    num_scenarios = params["num_scenarios"]
    scenarios = list(range(num_scenarios))

    # ---------- Uncertainty parameters ----------
    level = params["price_uncertainty"]
    if level == "Low":
        sigma_prices = 0.05
    elif level == "High":
        sigma_prices = 0.25
    else:
        sigma_prices = 0.15

    gamma_mean = {
        "Utility-scale PV":      0.85,
        "Onshore wind":          0.90,
        "Offshore wind (fixed)": 0.92,
        "OCGT - Natural gas":    1.05,
        "Diesel engine farm":    1.00,
        "Coal power plant":      1.00,
        "Nuclear power plant":   1.00,
    }

    # ---------- Generate scenarios ----------
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

    # ---------- Build + solve model ----------
    try:
        model, model_vars = _build_model(
            tech_data,
            co2_intensity,
            tech_names,
            time_periods,
            scenarios,
            params,
            hub_price_scenarios,
            gamma,
            sum_capacity_mw,
        )

        solver = pyo.SolverFactory("highs")
        results = solver.solve(model, tee=False)

    except Exception as e:
        return {"status": f"error: {e}"}

    # ---------- Check status ----------
    if (
        results.solver.status != SolverStatus.ok
        or results.solver.termxrination_condition
        not in (TerminationCondition.optimal, TerminationCondition.feasible)
    ):
        return {"status": "not_optimal"}

    # ---------- Extract results ----------
    results_dict = _extract_results(
        model,
        model_vars,
        tech_names,
        time_periods,
        scenarios,
        baseline_prices,
        hub_price_scenarios,
    )

    return results_dict


# ============================================================================
#                         SCENARIO GENERATION
# ============================================================================
def _generate_hub_price_scenarios(num_scenarios, baseline_prices, sigma=0.15):
    T = len(baseline_prices)
    scenarios = {}

    for s in range(num_scenarios):
        shock = 0.0
        price_path = []
        for t in range(T):
            eps = np.random.normal(0, sigma)
            shock = 0.7 * shock + eps
            price = baseline_prices[t] * (1 + shock)
            price = np.clip(price, -20, 300)
            price_path.append(price)

        scenarios[s] = price_path

    return scenarios


def _generate_capture_factor_scenarios(
    tech_names, gamma_mean, num_scenarios, num_periods, gamma_sigma=0.05, rho_gamma=0.7
):
    gamma = {}
    for tech in tech_names:
        for s in range(num_scenarios):
            shock = 0.0
            for t in range(num_periods):
                eps = np.random.normal(0, gamma_sigma)
                shock = rho_gamma * shock + eps
                g = gamma_mean.get(tech, 1.0) * (1 + shock)
                g = np.clip(g, 0.5, 1.3)
                gamma[(tech, s, t)] = g
    return gamma


# ============================================================================
#                            PYOMO MODEL
# ============================================================================
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

    initial_budget = params["initial_budget"]
    market_share = params["market_share"]
    demand_growth_rate = params["demand_growth_rate"]

    base_demand = market_share * 35_250_000
    carbon_price = 83.0
    carbon_increase = 5.0
    net_revenue_factor = 0.7

    # Economic parameters
    capex = {tech: tech_data[tech]["nominal_investment_total"] for tech in tech_names}
    elec_eff = {tech: tech_data[tech]["elec_eff"] for tech in tech_names}
    vom_meur = {tech: tech_data[tech]["variable_om_total"] / 1e6 for tech in tech_names}
    fixed_om_meur = {tech: tech_data[tech]["fixed_om_total"] / 1e6 for tech in tech_names}

    # Demand per period
    demand = {
        t: base_demand * ((1 + demand_growth_rate) ** t)
        for t in time_periods
    }

    carbon_price_meur_per_tco2 = {
        t: (carbon_price + carbon_increase*t) / 1e6
        for t in time_periods
    }

    # Prices → MEUR/MWh
    price_scenarios_meur = {
        s: [p/1e6 for p in hub_price_scenarios[s]]
        for s in scenarios
    }

    blocks = [1,2,3]
    pi_block = {}
    for s in scenarios:
        for t in time_periods:
            p = price_scenarios_meur[s][t]
            pi_block[(s,t,1)] = 1.00*p
            pi_block[(s,t,2)] = 0.85*p
            pi_block[(s,t,3)] = 0.60*p

    # Block limits
    Q_block = {}
    for t in time_periods:
        Q_block[(t,1)] = 0.25*demand[t]
        Q_block[(t,2)] = 0.35*demand[t]
        Q_block[(t,3)] = 0.40*demand[t]

    # Max build per tech
    max_capacity = {
        tech: market_share * sum_capacity_mw * 0.2
        for tech in tech_names
    }

    # -----------------------
    # Build Pyomo model
    # -----------------------
    m = pyo.ConcreteModel()

    # Sets
    m.TECH = pyo.Set(initialize=tech_names)
    m.T = pyo.Set(initialize=time_periods, ordered=True)
    m.S = pyo.Set(initialize=scenarios)
    m.B = pyo.Set(initialize=blocks)

    # Parameters
    m.capex = pyo.Param(m.TECH, initialize=capex, domain=pyo.NonNegativeReals)
    m.eff = pyo.Param(m.TECH, initialize=elec_eff, domain=pyo.NonNegativeReals)
    m.vom = pyo.Param(m.TECH, initialize=vom_meur, domain=pyo.NonNegativeReals)
    m.fom = pyo.Param(m.TECH, initialize=fixed_om_meur, domain=pyo.NonNegativeReals)
    m.co2_int = pyo.Param(m.TECH, initialize=co2_intensity, domain=pyo.NonNegativeReals)
    m.demand = pyo.Param(m.T, initialize=demand, domain=pyo.NonNegativeReals)
    m.carbon_price = pyo.Param(m.T, initialize=carbon_price_meur_per_tco2, domain=pyo.NonNegativeReals)
    m.max_cap = pyo.Param(m.TECH, initialize=max_capacity, domain=pyo.NonNegativeReals)

    m.pi_block = pyo.Param(m.S, m.T, m.B, initialize=pi_block)
    m.Q_block = pyo.Param(m.T, m.B, initialize=Q_block)

    # Capture factors
    m.gamma = pyo.Param(m.TECH, m.S, m.T, initialize=gamma)

    # Variables
    m.invest = pyo.Var(m.TECH, m.T, domain=pyo.NonNegativeReals)
    m.capacity = pyo.Var(m.TECH, m.T, domain=pyo.NonNegativeReals)

    m.energy = pyo.Var(m.TECH, m.T, m.S, domain=pyo.NonNegativeReals)
    m.sales = pyo.Var(m.TECH, m.T, m.B, m.S, domain=pyo.NonNegativeReals)

    m.budget = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)
    m.revenue = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)
    m.op_cost = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)
    m.co2_cost = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)
    m.fom_cost = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.capex_cost = pyo.Var(m.T,domain=pyo.NonNegativeReals)

    # -----------------------
    # Constraints
    # -----------------------
    m.cons = pyo.ConstraintList()

    # Capacity flow
    for tech in m.TECH:
        first_t = time_periods[0]
        m.cons.add(m.capacity[tech, first_t] == m.invest[tech, first_t])
        for t in time_periods[1:]:
            m.cons.add(
                m.capacity[tech, t] == m.capacity[tech, t-1] + m.invest[tech, t]
            )

        for t in time_periods:
            m.cons.add(m.capacity[tech, t] <= m.max_cap[tech])

    # Production limits
    for tech in m.TECH:
        for t in time_periods:
            for s in m.S:
                m.cons.add(
                    m.energy[tech, t, s] <=
                    m.capacity[tech, t] *
                    m.eff[tech] *
                    hours_per_year
                )

    # Costs
    for t in time_periods:

        m.cons.add(
            m.capex_cost[t] ==
            sum(m.invest[tech, t]*m.capex[tech] for tech in m.TECH)
        )

        m.cons.add(
            m.fom_cost[t] ==
            sum(m.capacity[tech, t]*m.fom[tech] for tech in m.TECH)
        )

        for s in m.S:
            m.cons.add(
                m.op_cost[t, s] ==
                sum(m.energy[tech, t, s]*m.vom[tech] for tech in m.TECH)
            )

            m.cons.add(
                m.co2_cost[t, s] ==
                sum(
                    m.energy[tech, t, s] *
                    m.co2_int[tech] *
                    m.carbon_price[t]
                    for tech in m.TECH
                )
            )

    # Sales = energy
    for tech in m.TECH:
        for t in time_periods:
            for s in m.S:
                m.cons.add(
                    sum(m.sales[tech, t, b, s] for b in m.B)
                    == m.energy[tech, t, s]
                )

    # Block constraints
    for t in time_periods:
        for b in m.B:
            for s in m.S:
                m.cons.add(
                    sum(m.sales[tech, t, b, s] for tech in m.TECH)
                    <= m.Q_block[t, b]
                )

    # Revenue
    for t in time_periods:
        for s in m.S:
            m.cons.add(
                m.revenue[t, s] ==
                sum(
                    m.sales[tech, t, b, s] *
                    m.pi_block[s, t, b] *
                    m.gamma[tech, s, t] *
                    net_revenue_factor
                    for tech in m.TECH
                    for b in m.B
                )
            )

    # Budget recursion
    for s in m.S:
        first_t = time_periods[0]
        m.cons.add(m.budget[first_t, s] == initial_budget)

        for t in time_periods:
            m.cons.add(m.capex_cost[t] <= m.budget[t, s])
            m.cons.add(m.budget[t, s] >= 0)

            if t < num_periods - 1:
                m.cons.add(
                    m.budget[t+1, s] ==
                    m.budget[t, s]
                    - m.capex_cost[t]
                    + m.revenue[t, s]
                    - m.op_cost[t, s]
                    - m.fom_cost[t]
                    - m.co2_cost[t, s]
                )

    # -----------------------
    # Objective
    # -----------------------
    S = len(scenarios)

    def obj_expr(m):
        return (1/S) * sum(
            m.revenue[t,s]
            - m.op_cost[t,s]
            - m.co2_cost[t,s]
            - m.fom_cost[t]
            - m.capex_cost[t]
            for t in time_periods
            for s in m.S
        )

    m.obj = pyo.Objective(rule=obj_expr, sense=pyo.maximize)

    return m, {
        "investment": m.invest,
        "capacity": m.capacity,
        "energy": m.energy,
        "sales": m.sales,
        "budget": m.budget,
        "revenue": m.revenue,
        "op_cost": m.op_cost,
        "co2_cost": m.co2_cost,
        "fixed_om_cost": m.fom_cost,
        "capex_cost": m.capex_cost,
        "time_periods": time_periods,
        "scenarios": scenarios,
        "tech_names": tech_names,
    }


# ============================================================================
#                       EXTRACT RESULTS (Pyomo)
# ============================================================================
def _extract_results(model, v, tech_names, time_periods, scenarios,
                     price_baseline, hub_price_scenarios):

    investment = v["investment"]
    capacity = v["capacity"]
    energy = v["energy"]
    revenue = v["revenue"]
    op_cost = v["op_cost"]
    co2_cost = v["co2_cost"]
    fixed_om_cost = v["fixed_om_cost"]
    capex_cost = v["capex_cost"]
    budget = v["budget"]

    # -------------------------------------
    # KPIs
    # -------------------------------------
    profit = []
    for s in scenarios:
        total = sum(
            value(revenue[t,s])
            - value(op_cost[t,s])
            - value(co2_cost[t,s])
            - value(fixed_om_cost[t])
            - value(capex_cost[t])
            for t in time_periods
        )
        profit.append(total)

    profit = np.array(profit)
    expected_profit = float(profit.mean())

    alpha = 0.10
    n_cvar = max(1, int(alpha * len(profit)))
    cvar10 = float(np.sort(profit)[:n_cvar].mean())

    final_budget = float(np.mean([value(budget[time_periods[-1], s]) for s in scenarios]))
    expected_revenue = float(np.mean([value(revenue[t, s]) for t in time_periods for s in scenarios]))

    cap_final = {tech: value(capacity[tech, time_periods[-1]]) for tech in tech_names}
    total_capacity = float(sum(cap_final.values()))

    # -------------------------------------
    # TABLES
    # -------------------------------------
    cap_table = pd.DataFrame(
        {
            f"Year {t}": {tech: value(capacity[tech, t]) for tech in tech_names}
            for t in time_periods
        }
    )
    cap_table.index.name = "Technology"
    cap_table.loc["Total"] = cap_table.sum(axis=0)
    cap_table = cap_table.round(1)

    prod_data = {}
    for tech in tech_names:
        prod_data[tech] = [
            np.mean([value(energy[tech, t, s]) for s in scenarios]) / 1e3
            for t in time_periods
        ]

    prod_table = pd.DataFrame(prod_data, index=time_periods).T
    prod_table.columns = [f"Year {t}" for t in time_periods]
    prod_table.loc["Total"] = prod_table.sum(axis=0)
    prod_table = prod_table.round(1)

    profit_table = pd.DataFrame(
        {"Scenario": scenarios, "Profit (MEUR)": profit}
    ).set_index("Scenario")
    profit_table = profit_table.round(1)

    economic = pd.DataFrame(
        {
            "Revenue": [
                np.mean([value(revenue[t,s]) for s in scenarios])
                for t in time_periods
            ],
            "Variable OPEX": [
                np.mean([value(op_cost[t,s]) for s in scenarios])
                for t in time_periods
            ],
            "Fixed O&M": [value(fixed_om_cost[t]) for t in time_periods],
            "CAPEX": [value(capex_cost[t]) for t in time_periods],
            "CO2 Cost": [
                np.mean([value(co2_cost[t, s]) for s in scenarios])
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

    # -------------------------------------
    # PLOTS (unchanged)
    # -------------------------------------
    plots = {}

    # ----------- capacity pie -----------
    cap_filtered = {tech: val for tech,val in cap_final.items() if val > 1e-6}

    fig1, ax = plt.subplots(figsize=(6,6))
    labels = list(cap_filtered.keys())
    sizes = list(cap_filtered.values())

    ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        startangle=90,
        colors=[TECH_COLORS.get(tech, "#CCCCCC") for tech in labels]
    )
    ax.set_title("Installed Capacity Mix (Final Year)")
    plots["capacity_pie"] = fig1

    # ----------- expected generation ---------
    fig4, ax4 = plt.subplots(figsize=(12,6))
    for tech in tech_names:
        mean_prod = np.array([
            np.mean([value(energy[tech, t, s]) for s in scenarios])
            for t in time_periods
        ]) / 1e3
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

    # ----------- financial flows ---------
    rev_vals = economic["Revenue"].values
    op_vals = economic["Variable OPEX"].values
    fix_vals = economic["Fixed O&M"].values
    capx_vals = economic["CAPEX"].values
    co2_vals = economic["CO2 Cost"].values
    net_vals = economic["Net Cashflow"].values

    x = np.arange(len(time_periods))

    fig3, ax3 = plt.subplots(figsize=(12,6))
    ax3.bar(x, rev_vals, label="Revenue", color="tab:green")
    ax3.bar(x, -op_vals, label="Variable OPEX", color="tab:orange")
    ax3.bar(x, -fix_vals, bottom=-op_vals, label="Fixed O&M", color="tab:gray")
    ax3.bar(
        x, -capx_vals,
        bottom=-op_vals-fix_vals,
        label="CAPEX",
        color="tab:blue",
    )
    ax3.bar(
        x, -co2_vals,
        bottom=-op_vals-fix_vals-capx_vals,
        label="CO₂ Cost",
        color="tab:red",
    )
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

    # ----------- price uncertainty ----------
    price_matrix = np.array([hub_price_scenarios[s] for s in scenarios])

    mean_price = price_matrix.mean(axis=0)
    min_price  = price_matrix.min(axis=0)
    max_price  = price_matrix.max(axis=0)
    p10 = np.percentile(price_matrix, 10, axis=0)
    p90 = np.percentile(price_matrix, 90, axis=0)

    fig_price, ax = plt.subplots(figsize=(10,5))
    ax.plot(time_periods, price_baseline, linestyle="--", color="black", label="Baseline Price")
    ax.plot(time_periods, mean_price, label="Scenario Mean", color="tab:blue")
    ax.fill_between(time_periods, min_price, max_price, alpha=0.2, label="Min–Max", color="tab:blue")
    ax.fill_between(time_periods, p10, p90, alpha=0.3, label="10–90% Band", color="tab:orange")
    ax.set_xlabel("Year")
    ax.set_ylabel("Hub Price (EUR/MWh)")
    ax.set_title("Stochastic Wholesale Price Scenarios")
    ax.grid(True)
    ax.legend()
    plots["price_uncertainty"] = fig_price

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