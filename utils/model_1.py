import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from utils.utils import TECH_COLORS


# -------------------------------------------------------------------
# Load data once — these are static for all runs
# -------------------------------------------------------------------
def load_technology_data():
    data_path = Path("data") / "technology_data.json"
    with data_path.open("r", encoding="utf-8") as f:
        technology_data = json.load(f)

    tech_units = technology_data["TECHNOLOGY_UNITS"]
    tech_data = technology_data["TECHNOLOGY_DATA"]

    return tech_units, tech_data


TECH_UNITS, TECH_DATA = load_technology_data()
TECH_NAMES = list(TECH_DATA.keys())


# -------------------------------------------------------------------
# Main function (Dashboard-compatible)
# -------------------------------------------------------------------
def run_model_1(params):
    """
    Runs the deterministic baseline model.
    Parameters expected in `params`:
        - budget
        - total_demand
        - min_num_technologies
        - min_share
    Returns a dict:
        - objective_value
        - tables (DataFrames)
        - plots (matplotlib figures)
    """

    # Unpack user parameters
    budget = params.get("budget", 500)
    total_demand = params.get("total_demand", 1_000_000)
    min_num_tech = params.get("min_num_technologies", 4)
    min_share = params.get("min_share_per_tech", 0.10)
    
    hours_per_year = 8760
    M = 1_000_000  # big-M
    
    # -----------------------------
    # Build model
    # -----------------------------
    model = gp.Model("Optimal_Plant_Mix")
    model.Params.LogToConsole = 0  # silence solver for dashboard use

    # Decision variables
    invest = model.addVars(TECH_NAMES, vtype=GRB.CONTINUOUS, name="Investment")
    energy = model.addVars(TECH_NAMES, vtype=GRB.CONTINUOUS, name="Energy")
    binary = model.addVars(TECH_NAMES, vtype=GRB.BINARY, name="Active")

    # -----------------------------
    # Constraints
    # -----------------------------

    # Budget constraint
    model.addConstr(
        gp.quicksum(invest[tech] * TECH_DATA[tech]["nominal_investment_total"]
                    for tech in TECH_NAMES) <= budget,
        "Budget"
    )

    # Match total energy requirement
    model.addConstr(
        gp.quicksum(energy[tech] for tech in TECH_NAMES) == total_demand,
        "EnergyMix"
    )

    # Minimum number of active technologies
    model.addConstr(
        gp.quicksum(binary[tech] for tech in TECH_NAMES) >= min_num_tech,
        "MinNumTech"
    )

    # Minimum share constraint
    for tech in TECH_NAMES:
        model.addConstr(
            energy[tech] >= min_share * total_demand * binary[tech],
            name=f"MinShare_{tech}"
        )

    # Production capacity constraint
    for tech in TECH_NAMES:
        eff = TECH_DATA[tech]["elec_eff"]
        model.addConstr(
            energy[tech] <= invest[tech] * hours_per_year * eff,
            name=f"CapLimit_{tech}"
        )

    # Link binary → energy using big-M
    for tech in TECH_NAMES:
        model.addConstr(
            energy[tech] <= M * binary[tech],
            name=f"BigM_{tech}"
        )

    # -----------------------------
    # Objective (Annualized CAPEX + VOM)
    # -----------------------------
    obj = (
        gp.quicksum(
            invest[tech]
            * (TECH_DATA[tech]["nominal_investment_total"]
               / TECH_DATA[tech]["technical_lifetime"])
            for tech in TECH_NAMES
        )
        +
        gp.quicksum(
            energy[tech] * (TECH_DATA[tech]["variable_om_total"] / 1_000_000)
            for tech in TECH_NAMES
        )
    )

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    # -----------------------------
    # Extract Results
    # -----------------------------
    if model.status != GRB.OPTIMAL:
        return {
            "status": "infeasible",
            "objective_value": None,
            "tables": {},
            "plots": {}
        }

    # Build results DataFrame
    df_results = pd.DataFrame({
        "Technology": TECH_NAMES,
        "Investment_MW": [invest[t].X for t in TECH_NAMES],
        "Energy_MWh": [energy[t].X for t in TECH_NAMES],
    })
    df_results["Share"] = df_results["Energy_MWh"] / total_demand
    df_results = df_results.sort_values("Share", ascending=False).reset_index(drop=True)

    # -----------------------------
    # Plots
    # -----------------------------

    # 1) Investment vs Energy Produced
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    colors = [TECH_COLORS.get(tech, "#CCCCCC") for tech in df_results["Technology"]]

    ax1.bar(
        df_results["Technology"],
        df_results["Investment_MW"],
        color=colors,
        label="Investment (MW)"
    )
    ax1.set_ylabel("Investment (MW)")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    for tech in df_results["Technology"]:
        ax2.scatter(
            tech,
            df_results.loc[df_results["Technology"] == tech, "Energy_MWh"],
            color="#1f77b4",
            s=80,               # optional: consistent marker size
            edgecolors="black", # optional: clearer dots
            zorder=5
        )

    ax2.set_ylabel("Energy Produced (MWh)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Technology Investment vs Energy Produced")
    fig1.tight_layout()

    # 2) Energy Distribution Pie Chart

    df_nonzero = df_results[df_results["Energy_MWh"] > 0]
    pie_colors = [
        TECH_COLORS.get(tech, "#CCCCCC")
        for tech in df_nonzero["Technology"]
    ]
    fig2, ax_pie = plt.subplots(figsize=(5, 5))
    ax_pie.pie(
        df_nonzero["Energy_MWh"],
        labels=df_nonzero["Technology"],
        autopct="%1.1f%%",
        startangle=90,
        colors=pie_colors
    )
    ax_pie.set_title("Energy Distribution by Technology")

    # -----------------------------
    # Return model results
    # -----------------------------
    return {
        "status": "optimal",
        "objective_value": model.objVal,
        "tables": {
            "results": df_results
        },
        "plots": {
            "investment_vs_energy": fig1,
            "energy_distribution": fig2
        }
    }