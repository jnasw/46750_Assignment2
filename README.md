# Assignment 2 – 46750 Optimisation in Modern Power Systems

## 1. Project Overview
This repository contains the implementation for **Assignment 2** of *Optimisation in Modern Power Systems*.  
The project develops three optimisation models of increasing complexity:

1. **Model 1 – Deterministic baseline**  
2. **Model 2 – Dynamic multi-period investment model**  
3. **Model 3 – Scenario-based and robustness model**

The repository includes all code required to reproduce the results presented in the written report, along with a **Streamlit dashboard** for interactive exploration of the developed models.

### Dashboard (Online Version)
**Deployed app:** *https://optimal-bidding-dashboard.streamlit.app*  
**Note:** Only **Models 1 and 2** run online. Model 3 requires a **local Gurobi academic license**, which cannot be activated on the cloud deployment. A local setup is described below.

## 2. Repository Structure

```text
├── data/
│   ├── electricity_prices.xlsx  
│   └── net_installed_capacity.csv
│   └── technology_data.json
│
├── utils/
│   ├── model_1.py              # model 1 for dashboard calls
│   ├── model_2.py              # model 2 for dashboard calls
│   ├── model_3_highs.py        # model 3 for dashboard calls, using pyomo and highs
│   ├── model_3.py              # model 3 for dashboard calls, using gurobi (standard)
│   └── utils.py                # data handling across models
│
├── docs/
│   ├── Assignment_2__2025.pdf  # Assignment description
│   └── consulting_proposal.pdf # Consulting proposal
│   └── Group20_report.pdf      # Final report
│
├── model_1.ipynb           # Deterministic baseline model
├── model_2.ipynb           # Dynamic multi-period investment model
├── model_3.ipynb           # Scenario-based & robustness model
├── model_comparison.ipynb  # Visualisations for the report
│
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 3 Installation & Environment Setup

**Python version:** 3.11.5 

### Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### Install project dependencies
```bash
pip install -r requirements.txt
```

A valid gurobi licence is needed to run the models. 

## 4. Running the Models

Each model is implemented as a standalone Jupyter notebook. The corresponding notebooks contain more information and documentation for each implementation step. 

## 5. Reproducing the Results (as in the Report)

To regenerate all results presented in the written report:

1. Run the notebooks in the following order:
   - **Model 1 → Model 2 → Model 3**
2. Inspect the resulting graphs and KPIs.

or

1. Run the Dashboard on the default settings
2. Inspect the resulting graphs and KPIs.

## 6. Running the Dashboard (Streamlit)

### Local execution
```bash
streamlit run app.py
```

The dashboard visualises:
- Installed capacity mixes
- Dispatch time series
- KPIs 
- Price and scenario uncertainty (Model 3 available locally only)



## 7. Gurobi License Setup

1. Request a free academic license:  
   https://www.gurobi.com/academia/academic-program-and-licenses/

2. Retrieve and install the license key:
   ```bash
   grbgetkey <YOUR-LICENSE-KEY>
   ```

3. Set the licence path
    ```bash
   export GRB_LICENSE_FILE=/path/to/gurobi.lic
   ```









