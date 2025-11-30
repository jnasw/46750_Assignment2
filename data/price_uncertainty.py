'''
Auxiliary Functions
'''
import pandas as pd
import numpy as np


def price_model(current_price, previous_price, mean_price):
    """
    Price process with dependence on previous prices.

    Args:
        current_price (float): Current electricity price.
        previous_price (float): Electricity price at the previous time step.
        data (dict): Fixed data containing model parameters.

    Returns:
        float: Next price.
    """

    mean_price = 90
    reversion_strength = 0.12
    price_cap = 120       # Increased price cap
    price_floor = -10      # Allow occasional negative prices

    mean_reversion = reversion_strength * (mean_price - current_price)
    noise = np.random.normal(0, 3)

    next_price = current_price + 0.6 * (current_price - previous_price) + mean_reversion + noise

    if next_price < 0:
        if np.random.rand() > 0.2:
            next_price = np.random.uniform(0, mean_price * 0.3)

    return max(min(next_price, price_cap), price_floor)


def generate_price_data(experiments):
    '''
    Generate price data for the simulation based on always the same starting parameters (mean values).

    Returns:
    '''
    price_data = []

    for e in range(experiments):
        # Initialize wind and price series with at least two values to enable stochastic functions
        price_series = [90, 90] # Start with current price values (according to the DEA)
        mean_price = 80
        
        # fill wind and price series by stochastic processes in functions
        num_timesteps = 20  # number of years
        for t in range(2, num_timesteps + 2):
            if t >= 2:
                price_series.append(price_model(price_series[t-1], price_series[t-2], mean_price))

        # Remove the two initial deterministic values
        price_series = price_series[2:]

        price_data.append(price_series)

    #save_wind_price_data_to_excel(wind_price_data)
    return price_data





