import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

from utils.utils import (
    load_technology_data,
    load_price_data,
    load_net_capacity_data,
    TECH_COLORS,
    HOURS_PER_YEAR,
)


# Load tech data once at import time
TECH_UNITS, TECH_DATA = load_technology_data()
TECH_NAMES = list(TECH_DATA.keys())


def _build_capacity_limits(market_share: float, max_capacity_share: float):
    """
    Build max capacity per technology (MW) based on market share and
    a max capacity share factor (fraction of total system capacity).
    """
    net_capacity_df, sum_capacity = load_net_capacity_data()

    max_capacity = {
        # "Gas turbine (simple cycle)": market_share * sum_capacity * max_capacity_share,
        # "Natural gas engine plant":   market_share * sum_capacity * max_capacity_share,
        "Diesel engine farm":         market_share * sum_capacity * max_capacity_share,
        "OCGT - Natural gas":         market_share * sum_capacity * max_capacity_share,
        "Coal power plant":           market_share * sum_capacity * max_capacity_share,
        "Nuclear power plant":        market_share * sum_capacity * max_capacity_share,
        "Onshore wind":               market_share * sum_capacity * max_capacity_share,
        "Offshore wind (fixed)":      market_share * sum_capacity * max_capacity_share,
        "Utility-scale PV":           market_share * sum_capacity * max_capacity_share,
    }

    return max_capacity


def _build_parameters(params):
    """
    Build all deterministic parameters for Model 2 from user inputs.
    """
    initial_budget = params.get("initial_budget", 100.0)
    market_share = params.get("market_share", 0.35)
    demand_growth_rate = params.get("demand_growth_rate", 0.02)
    num_periods = int(params.get("num_periods", 20))
    net_revenue_factor = params.get("net_revenue_factor", 0.7)
    max_capacity_share = params.get("max_capacity_share", 0.2)

    time_periods = list(range(num_periods))

    # Base demand as in original code
    base_demand = market_share * 35_250_000  # MWh

    demand = {
        t: base_demand * ((1 + demand_growth_rate) ** t)
        for t in time_periods
    }

    # Technical & economic parameters
    capex = {tech: TECH_DATA[tech]["nominal_investment_total"]
             for tech in TECH_NAMES}

    elec_eff = {tech: TECH_DATA[tech]["elec_eff"]
                for tech in TECH_NAMES}

    vom_eur = {tech: TECH_DATA[tech]["variable_om_total"]
               for tech in TECH_NAMES}
    vom_meur = {tech: vom_eur[tech] / 1e6 for tech in TECH_NAMES}

    fixed_om_eur = {tech: TECH_DATA[tech]["fixed_om_total"]
                    for tech in TECH_NAMES}
    fixed_om_meur = {tech: fixed_om_eur[tech] / 1e6 for tech in TECH_NAMES}

    # Prices
    df_prices = load_price_data(num_periods)
    price_list = df_prices["Price"].to_list()
    price_eur = {
        t: price_list[t] * net_revenue_factor
        for t in time_periods
    }
    price_meur = {t: price_eur[t] / 1e6 for t in time_periods}

    # Capacity limits
    max_capacity = _build_capacity_limits(
        market_share=market_share,
        max_capacity_share=max_capacity_share,
    )

    return {
        "initial_budget": initial_budget,
        "market_share": market_share,
        "demand_growth_rate": demand_growth_rate,
        "num_periods": num_periods,
        "time_periods": time_periods,
        "net_revenue_factor": net_revenue_factor,
        "demand": demand,
        "capex": capex,
        "elec_eff": elec_eff,
        "vom_meur": vom_meur,
        "fixed_om_meur": fixed_om_meur,
        "price_meur": price_meur,
        "max_capacity": max_capacity,
    }


def _build_and_solve_model(params_dict):
    """
    Build and solve the dynamic revenue-maximization model (deterministic).
    Returns the solved Gurobi model object.
    """
    time_periods = params_dict["time_periods"]
    demand = params_dict["demand"]
    capex = params_dict["capex"]
    elec_eff = params_dict["elec_eff"]
    vom_meur = params_dict["vom_meur"]
    fixed_om_meur = params_dict["fixed_om_meur"]
    price_meur = params_dict["price_meur"]
    max_capacity = params_dict["max_capacity"]
    initial_budget = params_dict["initial_budget"]
    num_periods = params_dict["num_periods"]

    model = gp.Model("Dynamic_Revenue_Maximization")
    model.Params.LogToConsole = 0  # quiet for dashboard

    # Variables
    investment = model.addVars(
        TECH_NAMES, time_periods,
        vtype=GRB.CONTINUOUS,
        name="Investment"
    )

    capacity = model.addVars(
        TECH_NAMES, time_periods,
        vtype=GRB.CONTINUOUS,
        name="Capacity"
    )

    energy = model.addVars(
        TECH_NAMES, time_periods,
        vtype=GRB.CONTINUOUS,
        name="Energy"
    )

    budget = model.addVars(
        time_periods,
        vtype=GRB.CONTINUOUS,
        name="Budget"
    )

    revenue = model.addVars(
        time_periods,
        vtype=GRB.CONTINUOUS,
        name="Revenue"
    )

    op_cost = model.addVars(
        time_periods,
        vtype=GRB.CONTINUOUS,
        name="OperatingCost"
    )

    fixed_om_cost = model.addVars(
        time_periods,
        vtype=GRB.CONTINUOUS,
        name="FixedOMCost"
    )

    capex_cost = model.addVars(
        time_periods,
        vtype=GRB.CONTINUOUS,
        name="CapexCost"
    )

    # Constraints

    # Capacity accumulation
    for tech in TECH_NAMES:
        model.addConstr(capacity[tech, 0] == investment[tech, 0],
                        name=f"CapacityInit_{tech}")
        for t in time_periods[1:]:
            model.addConstr(
                capacity[tech, t] == capacity[tech, t - 1] + investment[tech, t],
                name=f"CapacityAccum_{tech}_{t}"
            )

    # Cumulative capacity limits
    for tech in TECH_NAMES:
        if tech in max_capacity:
            for t in time_periods:
                model.addConstr(
                    capacity[tech, t] <= max_capacity[tech],
                    name=f"MaxCap_{tech}_{t}"
                )

    # Production limited by capacity * efficiency * hours
    for tech in TECH_NAMES:
        eff = elec_eff[tech]
        for t in time_periods:
            model.addConstr(
                energy[tech, t] <= capacity[tech, t] * eff * HOURS_PER_YEAR,
                name=f"ProdCap_{tech}_{t}"
            )

    # Revenue, OPEX, CAPEX, fixed O&M
    for t in time_periods:
        model.addConstr(
            revenue[t] == gp.quicksum(
                energy[tech, t] * price_meur[t] for tech in TECH_NAMES
            ),
            name=f"RevenueDef_{t}"
        )

        model.addConstr(
            op_cost[t] == gp.quicksum(
                energy[tech, t] * vom_meur[tech] for tech in TECH_NAMES
            ),
            name=f"OpCostDef_{t}"
        )

        model.addConstr(
            capex_cost[t] == gp.quicksum(
                investment[tech, t] * capex[tech] for tech in TECH_NAMES
            ),
            name=f"CapexCostDef_{t}"
        )

        model.addConstr(
            fixed_om_cost[t] == gp.quicksum(
                capacity[tech, t] * fixed_om_meur[tech] for tech in TECH_NAMES
            ),
            name=f"FixedOMDef_{t}"
        )

    # Budget dynamics
    model.addConstr(budget[0] == initial_budget, name="BudgetInit")

    for t in time_periods:
        # Cannot invest more than available budget
        model.addConstr(
            capex_cost[t] <= budget[t],
            name=f"InvLimit_{t}"
        )
        model.addConstr(
            budget[t] >= 0,
            name=f"BudgetNonNeg_{t}"
        )

        if t < num_periods - 1:
            model.addConstr(
                budget[t + 1] ==
                budget[t]
                - capex_cost[t]
                + revenue[t]
                - op_cost[t]
                - fixed_om_cost[t],
                name=f"BudgetDyn_{t}"
            )

    # Demand cap
    for t in time_periods:
        model.addConstr(
            gp.quicksum(energy[tech, t] for tech in TECH_NAMES) <= demand[t],
            name=f"DemandCap_{t}"
        )

    # Objective: maximize total profit over horizon
    model.setObjective(
        gp.quicksum(
            revenue[t] - op_cost[t] - fixed_om_cost[t] - capex_cost[t]
            for t in time_periods
        ),
        GRB.MAXIMIZE
    )

    model.optimize()

    return model


def _extract_results(model, params_dict):
    """
    Turn the solved model into KPIs, tables, and plots.
    """
    time_periods = params_dict["time_periods"]
    num_periods = params_dict["num_periods"]

    # Convenience references
    investment = model.getVars()

    # Rebuild variable dicts by name for clarity
    def v(name):
        return {var.varName: var for var in model.getVars() if var.varName.startswith(name)}

    # Build handle dicts (Gurobi doesn't auto-index them back)
    inv = {}
    cap = {}
    eng = {}
    bud = {}
    rev = {}
    op = {}
    fix = {}
    capex_cost = {}

    for var in model.getVars():
        if var.varName.startswith("Investment["):
            # Investment[tech,t]
            inside = var.varName[len("Investment["):-1]
            tech, t = inside.split(",")
            inv[(tech, int(t))] = var
        elif var.varName.startswith("Capacity["):
            inside = var.varName[len("Capacity["):-1]
            tech, t = inside.split(",")
            cap[(tech, int(t))] = var
        elif var.varName.startswith("Energy["):
            inside = var.varName[len("Energy["):-1]
            tech, t = inside.split(",")
            eng[(tech, int(t))] = var
        elif var.varName.startswith("Budget["):
            inside = var.varName[len("Budget["):-1]
            t = int(inside)
            bud[t] = var
        elif var.varName.startswith("Revenue["):
            inside = var.varName[len("Revenue["):-1]
            t = int(inside)
            rev[t] = var
        elif var.varName.startswith("OperatingCost["):
            inside = var.varName[len("OperatingCost["):-1]
            t = int(inside)
            op[t] = var
        elif var.varName.startswith("FixedOMCost["):
            inside = var.varName[len("FixedOMCost["):-1]
            t = int(inside)
            fix[t] = var
        elif var.varName.startswith("CapexCost["):
            inside = var.varName[len("CapexCost["):-1]
            t = int(inside)
            capex_cost[t] = var


    # ----- KPIs -----
    time_idx = time_periods
    total_profit = sum(rev[t].X - op[t].X - fix[t].X - capex_cost[t].X for t in time_idx)
    final_budget = bud[num_periods - 1].X
    total_revenue = sum(rev[t].X for t in time_idx)
    final_year = num_periods - 1
    total_capacity_end = sum(
        cap[(tech, final_year)].X
        for tech in TECH_NAMES
    )

    # ----- Tables -----
    capacity_df = pd.DataFrame({
        t: {tech: cap[(tech, t)].X for tech in TECH_NAMES}
        for t in time_idx
    })
    capacity_df.index.name = "Technology"
    capacity_df.loc["Total"] = capacity_df.sum(axis=0)

    production_df = pd.DataFrame({
        t: {tech: eng[(tech, t)].X / 1e3 for tech in TECH_NAMES}  # GWh
        for t in time_idx
    })
    production_df.index.name = "Technology"
    production_df.loc["Total"] = production_df.sum(axis=0)

    rev_cost_df = pd.DataFrame({
        t: {
            "Revenue (MEUR)": rev[t].X,
            "Var OPEX (MEUR)": op[t].X,
            "Fixed OPEX (MEUR)": fix[t].X,
            "CAPEX (MEUR)": capex_cost[t].X,
            "Net Cash Flow": rev[t].X - op[t].X - fix[t].X - capex_cost[t].X,
        }
        for t in time_idx
    })

    budget_df = pd.DataFrame({
        t: [bud[t].X] for t in time_idx
    }, index=["Budget (MEUR)"])

    totals_df = pd.DataFrame({
        "Total Revenue": [total_revenue],
        "Total Variable OPEX": [sum(op[t].X for t in time_idx)],
        "Total Fixed OPEX": [sum(fix[t].X for t in time_idx)],
        "Total CAPEX": [sum(capex_cost[t].X for t in time_idx)],
        "Final Budget": [final_budget],
        "Total Profit": [total_profit],
    })

    # ----- Plots -----

    # 1) Installed capacity over time (stackplot)
    cap_plot = {
        tech: [cap[(tech, t)].X for t in time_idx]
        for tech in TECH_NAMES
    }

    fig_cap = _plot_installed_capacity(time_idx, cap_plot)

    # 2) Financial flows (Revenue, OPEX, CAPEX, Net cash flow)
    revenue_series = [rev[t].X for t in time_idx]
    op_series = [op[t].X for t in time_idx]
    capex_series = [capex_cost[t].X for t in time_idx]

    fig_fin = _plot_financial_flows(time_idx, revenue_series, op_series, capex_series)


    return {
        "kpis": {
            "total_profit": total_profit,
            "final_budget": final_budget,
            "total_revenue": total_revenue,
            "total_capacity_end": total_capacity_end,
        },
        "tables": {
            "budget": budget_df.round(1),
            "capacity": capacity_df.round(1),
            "production": production_df.round(1),
            "financials": rev_cost_df.round(1),
            "totals": totals_df.round(1),
        },
        "plots": {
            "capacity_over_time": fig_cap,
            "financial_flows": fig_fin,
        },
    }


def _plot_installed_capacity(time_periods, capacity_dict, tech_names=TECH_NAMES):
    """
    Stacked area plot of installed capacity over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    cap_plot = {tech: [capacity_dict[tech,t].X for t in time_periods] 
            for tech in tech_names}

    display_periods = [t + 1 for t in time_periods]

    # Sort technologies by when they first get capacity
    def first_nonzero_period(tech):
        for i, val in enumerate(cap_plot[tech]):
            if val > 0:
                return i
        return float('inf')  # If never has capacity, put at end

    tech_list = sorted(cap_plot.keys(), key=first_nonzero_period)
    values = [capacity_dict[tech] for tech in tech_list]
    colors = [TECH_COLORS.get(tech, "#CCCCCC") for tech in tech_list]

    ax.stackplot(display_periods, values, labels=tech_list, colors=colors, alpha=0.9)

    ax.set_title("Installed Capacity Over Time")
    ax.set_xlabel("Year")
    ax.set_xticks(range(0, len(time_periods) + 2, 2))
    ax.set_ylabel("Capacity (MW)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


def _plot_financial_flows(time_periods, revenue, op_cost, capex_cost):
    """
    Plot revenue (positive) and OPEX + CAPEX (negative),
    all on a single shared y-axis, with net cashflow line.
    """
    rev = np.array(revenue)
    op = -np.array(op_cost)       # negative
    cap = -np.array(capex_cost)   # negative

    net_cash = rev + op + cap
    x = np.arange(len(time_periods))

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- BARS ---
    bar_width = 0.8

    ax.bar(x, rev, 
           label="Revenue", 
           color="tab:green", 
           width=bar_width)

    ax.bar(x, cap, 
           label="CAPEX (–)", 
           color="tab:blue", 
           width=bar_width)
    
    ax.bar(x, op, 
           label="OPEX (–)", 
           bottom=cap,  # Stack OPEX on top of CAPEX
           color="tab:orange", 
           width=bar_width)

    # Zero line
    ax.axhline(0, color="black", linewidth=1)

    # --- LINE ---
    ax.plot(
        x, net_cash,
        marker="o",
        linestyle="-",
        color="black",
        linewidth=2,
        label="Net Cashflow"
    )

    # Axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(time_periods)
    ax.set_xlabel("Year")
    ax.set_ylabel("MEUR")
    ax.set_title("Financial Flow per Period (MEUR)")
    ax.grid(True, axis="y", alpha=0.3)

    # --- LEGEND at bottom ---
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False
    )

    fig.tight_layout()
    return fig


def run_model_2(params):
    """
    Public API for Model 2 (for Streamlit):
      - Builds parameters from user inputs
      - Builds & solves the optimization model
      - Extracts KPIs, tables, plots

    Returns:
      {
        "status": "optimal" / "infeasible",
        "objective_value": float or None,
        "kpis": {...},
        "tables": {...},
        "plots": {...}
      }
    """
    params_dict = _build_parameters(params)
    model = _build_and_solve_model(params_dict)

    if model.status != GRB.OPTIMAL:
        return {
            "status": "infeasible",
            "objective_value": None,
            "kpis": {},
            "tables": {},
            "plots": {},
        }

    extracted = _extract_results(model, params_dict)

    return {
        "status": "optimal",
        "objective_value": model.objVal,
        **extracted,
    }