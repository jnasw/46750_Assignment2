# technology_data.py

TECHNOLOGY_UNITS = {
    "nominal_investment_total": "MEUR/MW_el",
    "fixed_om_total": "EUR/MW/year",
    "variable_om_total": "EUR/MWh",
    "elec_eff": "fraction (0-1)",
    "technical_lifetime": "years",
    "construction_time": "years",
    "total_outage": "fraction of time (0-1)"
}

TECHNOLOGY_DATA = {

    # ---- Fossil / Thermal ----
    'Gas turbine (simple cycle)': {
        'nominal_investment_total': 0.6,
        'fixed_om_total': 19778.73,
        'variable_om_total': 4.47,
        'elec_eff': 0.41,
        'technical_lifetime': 25,
        'construction_time': 1.5,
        'total_outage': 0.025
    },
    'Natural gas engine plant': {
        'nominal_investment_total': 0.5,
        'fixed_om_total': 6646.08,
        'variable_om_total': 6.38,
        'elec_eff': 0.48,
        'technical_lifetime': 25,
        'construction_time': 1,
        'total_outage': 0.01
    },
    'Diesel engine farm': {
        'nominal_investment_total': 0.36,
        'fixed_om_total': 8983.37,
        'variable_om_total': 6.38,
        'elec_eff': 0.35,
        'technical_lifetime': 25,
        'construction_time': 1,
        'total_outage': 0.01
    },
    'OCGT - Natural gas': {
        'nominal_investment_total': 0.47,
        'fixed_om_total': 8236.12,
        'variable_om_total': 4.79,
        'elec_eff': 0.41,
        'technical_lifetime': 25,
        'construction_time': 0.2,
        'total_outage': 0.01
    },
    'Coal power plant': {
        'nominal_investment_total': 2.1,
        'fixed_om_total': 34324.4,
        'variable_om_total': 3.21,
        'elec_eff': 0.52,
        'technical_lifetime': 25,
        'construction_time': 4.5,
        'total_outage': 0
    },

    # ---- Nuclear ----
    # from external source
    'Nuclear power plant': {
        'nominal_investment_total': 4,
        'fixed_om_total': 147700,
        'variable_om_total': 20,
        'elec_eff': 0.36,
        'technical_lifetime': 60,
        'construction_time': 8,
        'total_outage': 0.001
    },

    # ---- Wind ----
    'Onshore wind': {
        'nominal_investment_total': 1.15,
        'fixed_om_total': 16663,
        'variable_om_total': 1.98,
        'elec_eff': 0.41,
        'technical_lifetime': 30,
        'construction_time': 1.5,
        'total_outage': 0.023
    },
    'Offshore wind (fixed)': {
        'nominal_investment_total': 2.39,
        'fixed_om_total': 34000,
        'variable_om_total': 3.45,
        'elec_eff': 0.52,
        'technical_lifetime': 30,
        'construction_time': 3.5,
        'total_outage': 0.025
    },

    # ---- Solar ----
    'Utility-scale PV': {
        'nominal_investment_total': 0.38,
        'fixed_om_total': 9500,
        'variable_om_total': 0.5, # assumed since DEA Technology catalogue doesnt mention
        'elec_eff': 0.16,
        'technical_lifetime': 35,
        'construction_time': 0.5,
        'total_outage': None
    },
}


# plot graph to compare nominal investment costs
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    technologies = list(TECHNOLOGY_DATA.keys())
    investment_costs = [TECHNOLOGY_DATA[tech]['nominal_investment_total'] for tech in technologies]

    plt.figure(figsize=(10, 6))
    plt.barh(technologies, investment_costs, color='skyblue')
    plt.xlabel('Nominal Investment Cost (MEUR/MW_el)')
    plt.title('Nominal Investment Costs of Different Technologies')
    plt.grid(axis='x')
    plt.tight_layout()
    #plt.show()

# plot graph to compare variable O&M costs
    variable_om_costs = [TECHNOLOGY_DATA[tech]['variable_om_total'] for tech in technologies]
    plt.figure(figsize=(10, 6))
    plt.barh(technologies, variable_om_costs, color='salmon')
    plt.xlabel('Variable O&M Cost (EUR/MWh)')
    plt.title('Variable O&M Costs of Different Technologies')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()