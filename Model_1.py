import json
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path

# Define the path to the technology data JSON file
data_path = Path("data") / "technology_data.json"

# Load the JSON data
with data_path.open("r", encoding="utf-8") as f:
    technology_data = json.load(f)

# Extract the relevant parts of the data
TECHNOLOGY_UNITS = technology_data["TECHNOLOGY_UNITS"]
TECHNOLOGY_DATA = technology_data["TECHNOLOGY_DATA"]

# Convert the units data into a DataFrame
df_units = pd.DataFrame(list(TECHNOLOGY_UNITS.items()), columns=["Parameter", "Unit"])

# Convert the technology data into a DataFrame and transpose it for proper structure
df_tech = pd.DataFrame(TECHNOLOGY_DATA).T  # Transpose the DataFrame so each technology is a row
df_tech = df_tech.reset_index().rename(columns={"index": "Technology"})

# Merge the DataFrames based on the "Parameter" column
df = pd.merge(df_units, df_tech, left_on="Parameter", right_on="Technology", how="right").drop(columns=["Technology"])

# Define the initial budget (in MEUR)
initial_budget = 500  # The total available budget for investment in MW

# Assumed energy demand (in MWh), will not be used for meeting demand directly in this case
# Instead, we focus on energy mix selection based on budget and cost minimization
total_energy_mix = 1000000  # in MWh

# Define a large constant for big-M method
M = 1000000  # Large number to enforce energy production only when selected

# Hours in the year
hours_per_year = 8760

# Initialize the model
model = gp.Model("Optimal_Plant_Mix")

# Define decision variables for each technology
tech_names = list(TECHNOLOGY_DATA.keys())
investment_vars = model.addVars(tech_names, vtype=GRB.CONTINUOUS, name="Investment")

# Total installed capacity for each technology in MW (decision variable)
energy_produced_vars = model.addVars(tech_names, vtype=GRB.CONTINUOUS, name="EnergyProduced")

# Define binary variables to track if a technology is producing energy
binary_vars = model.addVars(tech_names, vtype=GRB.BINARY, name="BinaryProduced")

# Objective: Minimize the total annualized cost (CAPEX + O&M)
model.setObjective(
    gp.quicksum(
        investment_vars[tech] * (TECHNOLOGY_DATA[tech]['nominal_investment_total']/TECHNOLOGY_DATA[tech]['technical_lifetime']) for tech in tech_names
    ) +  # CAPEX cost in MEUR  --  devided by lifetime to annualize
    gp.quicksum(
        energy_produced_vars[tech] * (TECHNOLOGY_DATA[tech]['variable_om_total']/1000000) for tech in tech_names
    ),  # Variable O&M cost in MEUR
    GRB.MINIMIZE
)

# Constraints:

# 1. Ensure total annualized CAPEX stays within the initial budget
model.addConstr(
    gp.quicksum(
        investment_vars[tech] * TECHNOLOGY_DATA[tech]['nominal_investment_total'] for tech in tech_names
    ) <= initial_budget,  # Total investment must not exceed budget
    "Budget_Constraint"
)

# 2. Ensure the total energy produced by the selected technologies equals the required energy mix
model.addConstr(
    gp.quicksum(
        energy_produced_vars[tech] for tech in tech_names
    ) == total_energy_mix,  # Total energy produced should match the target energy mix
    "EnergyMix_Constraint"
)

# 3. Ensure at least 4 different technologies are selected
# Sum of binary variables should be at least 4
model.addConstr(
    gp.quicksum(
        binary_vars[tech] for tech in tech_names
    ) >= 4,  # At least 4 technologies must be producing energy
    "MinTechnologies_Constraint"
)

# 4. Ensure each selected technology produces at least 10% of the total energy mix
# Big-M method to link the energy production with the binary variable
model.addConstrs(
    (energy_produced_vars[tech] >= 0.1 * total_energy_mix * binary_vars[tech] for tech in tech_names)
    , name="MinTechnologyContribution"
)

# 5. The energy produced by each technology cannot exceed its installed capacity
model.addConstrs(
    (energy_produced_vars[tech] <= investment_vars[tech] * hours_per_year * TECHNOLOGY_DATA[tech]['elec_eff'] * 1000 for tech in tech_names)
    , name="MaxCapacity_Constraint"
)

# 6. Link binary variable with energy production (a binary variable is 1 if energy is produced, 0 otherwise)
# Using big-M to activate energy production only when binary variable is 1
model.addConstrs(
    (energy_produced_vars[tech] <= M * binary_vars[tech] for tech in tech_names)
    , name="BinaryLink"
)

# Solve the model
model.optimize()

# Output results
print("\n")

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"Total annual cost: {model.objVal:.2f} MEUR")
    print("Optimal plant mix (MW) and energy produced (MWh):")
    for tech in tech_names:
        if energy_produced_vars[tech].x > 0:  # Only print technologies that contribute to the energy mix
            print(f"{tech}: {investment_vars[tech].x:.4f} MW invested, {energy_produced_vars[tech].x:.2f} MWh produced, covering {(100*energy_produced_vars[tech].x/total_energy_mix):.2f}% of the demand")
else:
    print("Optimization did not succeed:", model.status)
