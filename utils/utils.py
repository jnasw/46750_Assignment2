from pathlib import Path
import json

import pandas as pd

# Common constants
HOURS_PER_YEAR = 8760

# Shared color scheme for technologies
TECH_COLORS = {
    "Gas turbine (simple cycle)": "#FF8C42",      # Orange (gas)
    "Natural gas engine plant":   "#E63946",      # Red (gas)
    "Diesel engine farm":         "#4A4A4A",      # Dark gray (diesel/oil)
    "OCGT - Natural gas":         "#FF6B9D",      # Pink (gas)
    "Coal power plant":           "#654321",      # Brown (coal)
    "Nuclear power plant":        "#9D4EDD",      # Purple (nuclear)
    "Onshore wind":               "#4A90E2",      # Blue (wind)
    "Offshore wind (fixed)":      "#2E5C8A",      # Dark blue (wind)
    "Utility-scale PV":           "#CFB105F1",      # Yellow/gold (solar),
}


def load_technology_data():
    """
    Load technology data JSON and return (units, data) dicts.
    """
    data_path = Path("data") / "technology_data.json"
    with data_path.open("r", encoding="utf-8") as f:
        technology_data = json.load(f)

    tech_units = technology_data["TECHNOLOGY_UNITS"]
    tech_data = technology_data["TECHNOLOGY_DATA"]

    return tech_units, tech_data


def load_price_data(num_periods: int):
    """
    Load electricity price time series from Excel, convert to EUR/MWh,
    and truncate to num_periods (max 20, based on original data).
    """
    price_data_path = Path("data") / "electricity_prices.xlsx"
    df_prices = pd.read_excel(price_data_path)[["Year", "Price"]].iloc[:num_periods]
    df_prices["Year"] = df_prices["Year"].astype(int)
    df_prices["Price"] = df_prices["Price"].astype(float)

    exchange_rate = 7.4  # DKK to EUR
    df_prices["Price"] = df_prices["Price"] / exchange_rate

    # Truncate to requested number of periods (max available)
    df_prices = df_prices.iloc[:num_periods].copy()

    return df_prices  # columns: Year, Price


def load_net_capacity_data():
    """
    Load net installed capacity CSV and return:
      - cleaned DataFrame
      - total installed capacity in MW (sum over all years and techs)
    """
    net_capacity = pd.read_csv(Path("data") / "net_installed_capacity.csv")

    # Drop metadata row, reset index
    net_capacity = net_capacity.iloc[1:].reset_index(drop=True)

    # Convert all columns except 'Year' to numeric (they are GW in the source)
    num_cols = [c for c in net_capacity.columns if c != "Year"]
    net_capacity[num_cols] = net_capacity[num_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Sum capacities (assume values are in GW) and convert to MW
    sum_capacity_gw = net_capacity[num_cols].sum().sum()
    sum_capacity_mw = sum_capacity_gw * 1e3  # MW

    return net_capacity, sum_capacity_mw

import pandas as pd

def load_co2_intensity(TECH_DATA, year=2030):
    """
    Loads CO₂ intensity values for each technology.

    Parameters:
        TECH_DATA : dict
            Dictionary from technology_data.json containing units and technical data.
        year : int
            Year to load PyPSA CO₂ intensities from.

    Returns:
        dict {tech: CO2_intensity_in_tCO2_per_MWh}
    """

    # Load PyPSA technology data
    url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{year}.csv"
    df = pd.read_csv(url, index_col=[0, 1])

    # Convert "/kW" to "/MW" where needed
    df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
    df.unit = df.unit.str.replace("/kW", "/MW")

    # Unstack so we get e.g. df.loc["gas"]["CO2 intensity"]
    df = df.value.unstack()

    co2_lookup = df["CO2 intensity"].to_dict()  # category → CO2 intensity

    # Map your technology names to PyPSA categories
    tech_map = {
        "Diesel engine farm":        "gas",
        "OCGT - Natural gas":        "gas",
        "Coal power plant":          "coal",
        "Nuclear power plant":       "nuclear",
        "Onshore wind":              "onwind",
        "Offshore wind (fixed)":     "offwind",
        "Utility-scale PV":          "solar-utility",
    }

    # Build final intensity dictionary for all techs
    co2_intensity = {}

    for tech in TECH_DATA.keys():
        if tech in tech_map:
            category = tech_map[tech]
            if category in co2_lookup:
                if tech == "Diesel engine farm":
                    co2_intensity[tech] = co2_lookup[category] * 1.2  # assume 20% higher than gas for diesel
                else:
                    co2_intensity[tech] = co2_lookup[category]
            else:
                co2_intensity[tech] = 0.0
        else:
            # default fallback for technologies not mapped
            co2_intensity[tech] = 0.0

    return co2_intensity