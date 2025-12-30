"""
battery_arbitrage_pyomo_mvp.py

MVP: Optimize battery charge/discharge against spot prices over a day.
- Linear program (LP)
- Variables: p_ch[t], p_dis[t], soc[t]
- Constraints: SOC dynamics, bounds, power limits
- Objective: maximize profit = sum(price * (dis - ch) * dt)

Input CSV columns: timestamp, price  (price in €/MWh)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyomo.environ as pyo


@dataclass(frozen=True)
class BatteryParams:
    # Energy capacity (MWh)
    e_max: float = 10.0
    e_min: float = 0.0

    # Power limits (MW)
    p_ch_max: float = 5.0
    p_dis_max: float = 5.0

    # Efficiencies
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    # Timestep length in hours (1.0 for hourly prices)
    dt_hours: float = 1.0

    # Initial SOC (MWh)
    soc0: float = 5.0

    # Optional terminal SOC requirement (MWh). If None, no terminal constraint.
    soc_terminal: Optional[float] = 5.0


def read_prices(csv_path: str) -> pd.Series:
    """
    Reads a CSV and returns a pd.Series of prices indexed 0..T-1.
    CSV required columns: timestamp, price
    """
    df = pd.read_csv(csv_path)
    if "Day-ahead Price (EUR/MWh)" not in df.columns:
        raise ValueError("CSV must contain a 'price' column.")
    # timestamp is optional for optimization; nice for later plotting
    prices = df["Day-ahead Price (EUR/MWh)"].astype(float).reset_index(drop=True)
    if len(prices) == 0:
        raise ValueError("No prices found in CSV.")
    return prices


def build_and_solve(prices: pd.Series, params: BatteryParams, solver_name: str = "highs"):
    """
    Builds the Pyomo model and solves it.
    Returns: (model, results_df)
    """
    T = len(prices)
    dt = params.dt_hours

    if params.eta_ch <= 0 or params.eta_ch > 1:
        raise ValueError("eta_ch must be in (0, 1].")
    if params.eta_dis <= 0 or params.eta_dis > 1:
        raise ValueError("eta_dis must be in (0, 1].")
    if params.e_min < 0 or params.e_min > params.e_max:
        raise ValueError("Check e_min/e_max.")
    if not (params.e_min <= params.soc0 <= params.e_max):
        raise ValueError("soc0 must be within [e_min, e_max].")

    m = pyo.ConcreteModel("BatteryArbitrageMVP")

    # Time index: 0..T-1 for actions; SOC is defined 0..T (includes initial SOC at t=0)
    m.T = pyo.RangeSet(0, T - 1)
    m.S = pyo.RangeSet(0, T)

    # Parameters
    price_dict = {t: float(prices.iloc[t]) for t in range(T)}
    m.price = pyo.Param(m.T, initialize=price_dict, within=pyo.Reals)

    # Decision variables
    m.p_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)   # MW
    m.p_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # MW
    m.soc = pyo.Var(m.S, domain=pyo.Reals)               # MWh

    # Bounds
    def soc_bounds_rule(m, s):
        return (params.e_min, m.soc[s], params.e_max)
    m.soc_bounds = pyo.Constraint(m.S, rule=soc_bounds_rule)

    def pch_bounds_rule(m, t):
        return m.p_ch[t] <= params.p_ch_max
    m.pch_bounds = pyo.Constraint(m.T, rule=pch_bounds_rule)

    def pdis_bounds_rule(m, t):
        return m.p_dis[t] <= params.p_dis_max
    m.pdis_bounds = pyo.Constraint(m.T, rule=pdis_bounds_rule)

    # Initial SOC
    m.soc_init = pyo.Constraint(expr=m.soc[0] == params.soc0)

    # SOC dynamics
    # soc[t+1] = soc[t] + (eta_ch * p_ch[t] - (1/eta_dis)*p_dis[t]) * dt
    def soc_dyn_rule(m, t):
        return m.soc[t + 1] == m.soc[t] + (params.eta_ch * m.p_ch[t] - (1.0 / params.eta_dis) * m.p_dis[t]) * dt
    m.soc_dyn = pyo.Constraint(m.T, rule=soc_dyn_rule)

    # Optional terminal SOC constraint (helps avoid "end of horizon dump" behavior)
    if params.soc_terminal is not None:
        if not (params.e_min <= params.soc_terminal <= params.e_max):
            raise ValueError("soc_terminal must be within [e_min, e_max].")
        m.soc_terminal = pyo.Constraint(expr=m.soc[T] >= params.soc_terminal)

    # Objective: maximize profit
    # Profit_t = price[t] * (p_dis[t] - p_ch[t]) * dt
    m.obj = pyo.Objective(
        expr=sum(m.price[t] * (m.p_dis[t] - m.p_ch[t]) * dt for t in m.T),
        sense=pyo.maximize
    )

    # Solve
    solver = pyo.SolverFactory(solver_name)
    if solver is None or not solver.available():
        raise RuntimeError(
            f"Solver '{solver_name}' not available.\n"
            f"Try installing HiGHS: pip install highspy\n"
            f"Or use another solver (cbc, glpk, gurobi, cplex) and pass its name."
        )

    res = solver.solve(m, tee=False)

    # Collect results
    out = pd.DataFrame({
        "t": list(range(T)),
        "price": [price_dict[t] for t in range(T)],
        "p_charge_MW": [pyo.value(m.p_ch[t]) for t in range(T)],
        "p_discharge_MW": [pyo.value(m.p_dis[t]) for t in range(T)],
        "soc_MWh_start": [pyo.value(m.soc[t]) for t in range(T)],
        "soc_MWh_end": [pyo.value(m.soc[t + 1]) for t in range(T)],
    })
    out["profit_EUR"] = out["price"] * (out["p_discharge_MW"] - out["p_charge_MW"]) * dt
    total_profit = out["profit_EUR"].sum()

    return m, res, out, total_profit


def main():
    if len(sys.argv) < 2:
        print("Usage: python battery_arbitrage_pyomo_mvp.py path/to/prices.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    # --- tweak these for your MVP ---
    params = BatteryParams(
        e_max=10.0,
        e_min=0.0,
        p_ch_max=5.0,
        p_dis_max=5.0,
        eta_ch=0.95,
        eta_dis=0.95,
        dt_hours=1.0,
        soc0=5.0,
        soc_terminal=5.0,  # set None to remove terminal constraint
    )

    prices = read_prices(csv_path)
    model, solver_results, results_df, total_profit = build_and_solve(prices, params, solver_name="highs")

    print("\n=== Solve status ===")
    print(solver_results.solver.status)
    print(solver_results.solver.termination_condition)

    print(f"\nTotal profit (EUR): {total_profit:.2f}")
    print(results_df)
    # Save results for plotting/backtest

    results_df['soc_percent'] = results_df['soc_MWh_start'] / params.e_max * 100.0
    results_df.to_csv('data/dispatch_results.csv', index=False)
    print("Wrote dispatch_results.csv")


if __name__ == "__main__":
    main()
